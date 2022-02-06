import logging
import os.path
import sys
import time
import uuid
import winsound
from datetime import datetime

import cv2
from azure.storage.blob import BlobServiceClient

EMPTY_FRAMES_THRESHOLD = 50
MAX_CACHE_SIZE = 50

FACE_CASCADE_PATH = "resources\\detection_cascades\\haarcascade_frontalcatface.xml"
BASE_OUTPUT_PATH = "resources\\images\\"
BLOB_CONTAINER_NAME = "accessmonitorblob"
CAMERA_ID = "camera_01"
AZURESTORAGE_CONNECTION_STRING = os.getenv("AZURESTORAGE_CONNECTION_STRING")
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def initialize_cascade(cascade_path):
    cascade = cv2.CascadeClassifier()
    if not cascade.load(cv2.samples.findFile(cascade_path)):
        print(f"Error during cascade loading {cascade_path}")
        exit(0)
    return cascade


def setup_videocapture(device_id=0, resolution_width=1280, resolution_height=720):
    video = cv2.VideoCapture(device_id)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, resolution_width)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution_height)
    return video


def upload_detected(file_full_path: str, blob_filename: str):
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURESTORAGE_CONNECTION_STRING)
        blob_client = blob_service_client.get_blob_client(container=BLOB_CONTAINER_NAME,
                                                          blob=os.path.join(CAMERA_ID, blob_filename))
        with open(file_full_path, "rb") as file:
            blob_client.upload_blob(file.read())

            winsound.Beep(2500, 1000)
            logging.debug(f"Uploaded detection {file_full_path}")
    finally:
        os.remove(file_full_path)
        logging.debug(f"Cleaned up local storage from detection {file_full_path}")


def detect(cascade, video_capture):
    cached_frames = {}
    empty_frames_count = 0
    while True:
        ret, frame_img = video_capture.read()
        gray_frame = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.equalizeHist(gray_frame)
        faces = len(cascade.detectMultiScale(gray_frame, minSize=[70, 70], flags=cv2.CASCADE_SCALE_IMAGE))
        laplace = cv2.Laplacian(gray_frame, cv2.CV_64F).var()

        if faces and detection_cache_is_not_full(cached_frames):
            cached_frames.update({laplace: frame_img})
            logging.debug(
                f"Added frame to cache - empty frames count: {empty_frames_count} - cached frames: {len(cached_frames)}")
            empty_frames_count = 0
            # time.sleep(3)

        if not faces:
            empty_frames_count += 1
            logging.debug(
                f"Detected empty frame - empty frames count: {empty_frames_count} - cached frames: {len(cached_frames)}")

        if (not_detecting_faces_since(empty_frames_count) and there_are_some_cached_detections(
                cached_frames)) or detection_cache_is_full(cached_frames):
            logging.debug(f"Attempting to write to blob faces:{faces} - cached frames: {len(cached_frames)}")

            blob_filename = create_blob_filename()
            full_file_path = os.path.join(BASE_OUTPUT_PATH, blob_filename)
            cv2.imwrite(full_file_path, cached_frames[max(cached_frames)])
            upload_detected(full_file_path, blob_filename)

            cached_frames.clear()
            empty_frames_count = 0
            time.sleep(5)


def create_blob_filename():
    filename = f"{str(uuid.uuid4())}_{datetime.today().strftime('%Y-%m-%d-%H-%M-%S')}.jpg"
    return filename


def detection_cache_is_not_full(cached_frames):
    return len(cached_frames) < MAX_CACHE_SIZE


def detection_cache_is_full(cached_frames):
    return len(cached_frames) >= MAX_CACHE_SIZE


def there_are_some_cached_detections(cached_frames):
    return len(cached_frames)


def not_detecting_faces_since(empty_frames_count):
    return empty_frames_count >= EMPTY_FRAMES_THRESHOLD


cascade = initialize_cascade(cascade_path=FACE_CASCADE_PATH)
logging.info(f"Initialized haar cascade {cascade}")

capture = setup_videocapture(device_id=0, resolution_width=1280, resolution_height=720)
logging.info(f"Initialized video capture {capture}")

detect(cascade=cascade, video_capture=capture)
