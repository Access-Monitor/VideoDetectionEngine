import os.path
import time
import uuid
from datetime import datetime

import cv2

from azure.storage.blob import BlobServiceClient

FACE_CASCADE_PATH = "resources\\detection_cascades\\haarcascade_frontalcatface.xml"
DEFAULT_OUTPUT_PATH = "resources\\images\\"
BLOB_CONTAINER_NAME = "accessmonitorblob"
AZURESTORAGE_CONNECTION_STRING = os.getenv("AZURESTORAGE_CONNECTION_STRING")


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
    blob_service_client = BlobServiceClient.from_connection_string(AZURESTORAGE_CONNECTION_STRING)
    blob_client = blob_service_client.get_blob_client(container=BLOB_CONTAINER_NAME, blob=blob_filename)
    with open(os.path.join(DEFAULT_OUTPUT_PATH, blob_filename), "rb") as file:
        img_data = file.read()
    blob_client.upload_blob(img_data)


def detect(cascade, video_capture):
    while True:
        ret, frame_img = video_capture.read()
        gray_frame = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.equalizeHist(gray_frame)
        faces = cascade.detectMultiScale(gray_frame, minSize=[70, 70], flags=cv2.CASCADE_SCALE_IMAGE)

        if len(faces):
            filename = f"{str(uuid.uuid4())}_{datetime.today().strftime('%Y-%m-%d-%H-%M-%S')}.jpg"
            cv2.imwrite(os.path.join(DEFAULT_OUTPUT_PATH, filename), frame_img)
            upload_detected(filename)
            time.sleep(15)


cascade = initialize_cascade(cascade_path=FACE_CASCADE_PATH)
capture = setup_videocapture(device_id=1, resolution_width=1280, resolution_height=720)
detect(cascade=cascade, video_capture=capture)
