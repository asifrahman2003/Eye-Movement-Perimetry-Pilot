"""
    @program: gaze_estimate.py
    @description: using mediapipe face mesh to extract per-frame gaze estimates (yaw, pitch)
                  from a recorded video, then saves the time series to CSV. 
"""

# required libraries
import cv2
import pandas as pd
import mediapipe as mp
from scipy.signal import find_peaks

# initializes mediapipe face mesh for iris and face landmark detection
mpfm = mp.solutions.face_mesh
cap  = cv2.VideoCapture("/Users/asifrahman/Documents/gazeDetectionTest.mp4")

yaws, pitches, times = [], [], []
fps = cap.get(cv2.CAP_PROP_FPS)
frame_idx = 0

def compute_gaze_from_landmarks(lm):
    # computing normalized gaze offsets (horizontal and vertical)
    # using iris midpoint and eye corner midpoint
    # Left eye corners: lm[33], lm[133]
    # Left iris points: lm[474–478]
    left_corner  = lm[33]       # left eye inner corner
    right_corner = lm[133]      # left eye outer corner
    iris_pts     = [lm[i] for i in range(474, 478)]     # iris landmarks for left eye


    # compute average of iris landmark coordinates
    iris_x = sum([p.x for p in iris_pts]) / len(iris_pts)
    iris_y = sum([p.y for p in iris_pts]) / len(iris_pts)
    iris_center = type(iris_pts[0])(x=iris_x, y=iris_y, z=0)


    # horizontal offset normalized to eye width
    eye_width = right_corner.x - left_corner.x
    horiz_offset = (iris_center.x - (left_corner.x + eye_width/2)) / eye_width


    # vertical offset similarly (using y coords):
    vert_offset = (iris_center.y - (left_corner.y + (lm[159].y - lm[33].y)/2)) / (lm[159].y - lm[33].y)


    # Convert offsets into “degrees” or keep as unitless for peak detection
    return horiz_offset, vert_offset

# process video frame by frame to detect gaze
with mpfm.FaceMesh(static_image_mode=False, refine_landmarks=True, max_num_faces=1) as fm:
    while True:
        # read frame
        ret, frame = cap.read()
        if not ret:
            break

        # converts BG to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # run face mesh
        res = fm.process(rgb)

        # compute gaze or set None if no face is deetected
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            yaw, pitch = compute_gaze_from_landmarks(lm)
        else:
            yaw = pitch = None

        # record yaw, pitch, and timestamp
        yaws.append(yaw)
        pitches.append(pitch)
        times.append(frame_idx / fps)
        frame_idx += 1

cap.release()

# save gaze time seriees to .csv file
df = pd.DataFrame({"time_s": times, "yaw": yaws, "pitch": pitches})
df.to_csv("mediapipe_gaze.csv", index=False)

df = pd.read_csv("mediapipe_gaze.csv")
# detecting yaw peaks in the saved csv for stimulus analysis
peaks, _ = find_peaks(df.yaw.fillna(0), distance=0.8*fps)  
peak_times = df.time_s.iloc[peaks]