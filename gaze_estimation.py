#Code for deep learning eye movement perimetry

# details for configuration
API_KEY= "rf_ANRREicJZoN6joOyLY2NVlQ2EFX2"
VIDEO_PATH = "/Users/asifrahman/Documents/gazeDetectionTest.mov"
OUTPUT_CSV = "gaze_output.csv"

# gaze detection URL
GAZE_DETECTION_URL = (
    "http://127.0.0.1:9001/gaze/gaze_detection?api_key=" + API_KEY
)

import os
import cv2
import numpy as np
import pandas as pd
import base64
import requests

# load all frames from the video
frames = []
video = cv2.VideoCapture(VIDEO_PATH)

while True:
    read, frame= video.read()
    if not read:
        break
    frames.append(frame)
frames = np.array(frames)

frameWidth = video.get(cv2.CAP_PROP_FRAME_WIDTH)
frameHeight = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = video.get(cv2.CAP_PROP_FPS)
print(frameWidth, frameHeight, fps)

# call the gaze-detection API on each frame
def detect_gazes(image):
    img_encode = cv2.imencode(".jpg", image)[1]
    img_base64 = base64.b64encode(img_encode)
    resp = requests.post(
        GAZE_DETECTION_URL,
        json={
            "api_key": API_KEY,
            "image": {"type": "base64", "value": img_base64.decode("utf-8")},
        },
    )
    if (len(resp.json()[0]["predictions"])) > 0:
        yaw = resp.json()[0]["predictions"][0]["yaw"]
        pitch = resp.json()[0]["predictions"][0]["pitch"]
        return yaw, pitch

yaw_data = []
pitch_list = []

for frame in frames:
    yaw, pitch = detect_gazes(frame)
    yaw_data.append(yaw)
    pitch_list.append(pitch)    
    
yaw_array = np.array(yaw_data)
pitch_array = np.array(pitch_list)

df = pd.DataFrame()
df["yaw"] = yaw_array
df["pitch"] = pitch_array

df.to_csv(OUTPUT_CSV)
