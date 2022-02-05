import logging
import os.path
import sys
import time
import uuid
import winsound
from datetime import datetime

import cv2
from azure.storage.blob import BlobServiceClient

MAX_CACHE_SIZE = 1000

FACE_CASCADE_PATH = "resources\\detection_cascades\\haarcascade_frontalcatface.xml"
BASE_OUTPUT_PATH = "resources\\images\\"
BLOB_CONTAINER_NAME = "accessmonitorblob"
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


def upload_detected(blob_filename):
    file_path = os.path.join(BASE_OUTPUT_PATH, blob_filename)
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURESTORAGE_CONNECTION_STRING)
        blob_client = blob_service_client.get_blob_client(container=BLOB_CONTAINER_NAME, blob=blob_filename)
        with open(file_path, "rb") as file:
            # blob_client.upload_blob(file.read())

            winsound.Beep(2500, 1000)
            logging.debug(f"Uploaded detection {file_path}")
    finally:
        os.remove(file_path)
        logging.debug(f"Cleaned up local storage from detection {file_path}")


def detect(cascade, video_capture):
    cached_frames = {}
    empty_frames_count = 0
    while True:
        ret, frame_img = video_capture.read()
        gray_frame = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.equalizeHist(gray_frame)
        faces = len(cascade.detectMultiScale(gray_frame, minSize=[70, 70], flags=cv2.CASCADE_SCALE_IMAGE))
        laplace = cv2.Laplacian(gray_frame, cv2.CV_64F).var()

        if faces and len(cached_frames) < MAX_CACHE_SIZE:
            cached_frames.update({laplace: frame_img})
            logging.debug(
                f"Added frame to cache - empty frames count: {empty_frames_count} - cached frames: {len(cached_frames)}")
            empty_frames_count = 0
            # time.sleep(3)

        if not faces:
            empty_frames_count += 1
            logging.debug(
                f"Detected empty frame - empty frames count: {empty_frames_count} - cached frames: {len(cached_frames)}")

        if (empty_frames_count >= 100 and len(cached_frames)) or len(cached_frames) >= MAX_CACHE_SIZE:
            logging.debug(f"Attempting to write to blob faces:{faces} - cached frames: {len(cached_frames)}")
            filename = f"{str(uuid.uuid4())}_{datetime.today().strftime('%Y-%m-%d-%H-%M-%S')}.jpg"
            cv2.imwrite(os.path.join(BASE_OUTPUT_PATH, filename), cached_frames[max(cached_frames)])
            upload_detected(filename)

            cached_frames.clear()
            empty_frames_count = 0
            # time.sleep(5)


cascade = initialize_cascade(cascade_path=FACE_CASCADE_PATH)
logging.info(f"Initialized haar cascade {cascade}")

capture = setup_videocapture(device_id=0, resolution_width=1280, resolution_height=720)
logging.info(f"Initialized video capture {capture}")

detect(cascade=cascade, video_capture=capture)
