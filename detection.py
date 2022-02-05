import logging
import os.path
import time
import uuid
import winsound
from datetime import datetime

import cv2
from azure.storage.blob import BlobServiceClient

FACE_CASCADE_PATH = "resources\\detection_cascades\\haarcascade_frontalcatface.xml"
BASE_OUTPUT_PATH = "resources\\images\\"
BLOB_CONTAINER_NAME = "accessmonitorblob"
AZURESTORAGE_CONNECTION_STRING = os.getenv("AZURESTORAGE_CONNECTION_STRING")
log = logging.getLogger("detectionLogger")


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
            img_data = file.read()
            blob_client.upload_blob(img_data)
            winsound.Beep(2500, 1000)
            log.debug(f"Uploaded detection {file_path}")
    finally:
        os.remove(file_path)
        log.debug(f"Cleaned up local storage from detection {file_path}")


def detect(cascade, video_capture):
    while True:
        ret, frame_img = video_capture.read()
        gray_frame = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.equalizeHist(gray_frame)
        faces = cascade.detectMultiScale(gray_frame, minSize=[70, 70], flags=cv2.CASCADE_SCALE_IMAGE)

        if len(faces):
            log.debug("Detected face from video record")
            filename = f"{str(uuid.uuid4())}_{datetime.today().strftime('%Y-%m-%d-%H-%M-%S')}.jpg"
            cv2.imwrite(os.path.join(BASE_OUTPUT_PATH, filename), frame_img)
            upload_detected(filename)
            time.sleep(30)


cascade = initialize_cascade(cascade_path=FACE_CASCADE_PATH)
log.info(f"Initialized haar cascade {cascade}")

capture = setup_videocapture(device_id=0, resolution_width=1280, resolution_height=720)
log.info(f"Initialized video capture {capture}")

detect(cascade=cascade, video_capture=capture)
