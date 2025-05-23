# Eye-Movement-Perimetry-Pilot
This repository contains a complete, offline pipeline for estimating eye gaze direction (yaw, pitch) from a recorded video of a moving black‑dot stimulus, detecting gaze shifts, and quantifying their alignment with a known 3s stimulus schedule.

## Repository Structure
```
├── gaze_estimate.py         # Extract per-frame yaw/pitch via MediaPipe Face Mesh
├── gaze_analysis.py         # Smooth signals, detect peaks, compute metrics, save plots
├── gaze_metrics.py          # Compute SRT, deviation, directional detection, and generate heatmaps
├── mediapipe_gaze.csv       # Raw time series: time_s, yaw, pitch
├── gaze_peaks.csv           # Stimulus-to-peak mappings: stim_time_s, yaw_peak_s, yaw_err_s, pitch_peak_s, pitch_err_s
├── srt_results.csv          # Yaw/pitch Saccadic Reaction Time values
├── deviation_results.csv    # Gaze deviation and detection success per stimulus
├── deviation_summary_by_direction.csv  # Grouped deviation stats
├── analysis_outputs/        # Saved plot images
│   ├── yaw_plot.png         # Yaw vs. time with detected peaks and stimulus markers
│   ├── pitch_plot.png       # Pitch vs. time with detected peaks and stimulus markers
│   ├── visual_field_heatmap.png         # Heatmap of directional deviation
│   ├── visual_field_detection_map.png   # Heatmap of detection accuracy
│   └── visual_field_srt_heatmap.png     # Heatmap of SRT values
├── GazeEstimatorReport.pdf  # Detailed documentation, methods, results, discussion
└── README.md                # This file
```

## Getting Started
### Prerequisites

- Python 3.7+  
- pip (Python package manager)  

### Installation

1. **Clone the repository**  
   ```
   git clone https://github.com/asifrahman2003/Eye-Movement-Perimetry-Pilot.git
   cd Eye-Movement-Perimetry-Pilot
   ```

### Install required Python packages:
```pip install mediapipe opencv-python pandas numpy scipy matplotlib```

## Usage

### 1. Gaze Extraction
Runs MediaPipe Face Mesh on the recorded video to extract normalized yaw and pitch per frame.
```python gaze_estimate.py --video path/to/your/video.mp4```
Outputs mediapipe_gaze.csv in the working directory.

### 2. Gaze Analysis

Loads the raw gaze CSV, smooths the signals, detects gaze-shift peaks, matches them to the 3 s stimulus schedule, computes accuracy metrics, and saves plots.
```python gaze_analysis.py```
Outputs:
gaze_peaks.csv (stimulus-to-peak details)
analysis_outputs/yaw_plot.png
analysis_outputs/pitch_plot.png

### 3. Gaze Metrics Analysis

Compute directional gaze accuracy, SRT (reaction time), and create clinical-style heatmaps.
```python gaze_metrics.py```

Outputs:
- srt_results.csv
- deviation_results.csv
- analysis_outputs/visual_field_heatmap.png
- analysis_outputs/visual_field_detection_map.png
- analysis_outputs/visual_field_srt_heatmap.png


## Methodology Summary

Landmark-Based Gaze Estimation
Use MediaPipe Face Mesh to detect 468 facial landmarks (including iris points) per frame.
Compute normalized horizontal (yaw) and vertical (pitch) offsets from eye corners and iris center.
Save a time-series CSV: time_s, yaw, pitch.
Signal Processing
Interpolate missing values and apply a 5‑frame moving average to smooth noise.
Estimate video FPS from timestamps and generate ideal stimulus times every 3s.
Peak Detection & Matching
Detect local maxima in the smoothed yaw and pitch signals using scipy.signal.find_peaks.
Match each 3s stimulus to the nearest detected peak within ±0.5s tolerance.
Calculate accuracy (% stimuli captured) and mean timing error.
Directional Error Thresholding & Visual Field Mapping
Apply directional error thresholds to classify gaze alignment success.
Generate visual field maps illustrating directional deviation, detection accuracy, and saccadic reaction times.

## Visualization
Plot yaw and pitch vs. time, overlaying detected peaks (×) and stimulus markers (dashed lines).
Heatmaps of directional deviation, detection accuracy, and SRT for each stimulus direction

## Original vs. Final Approach

Original Code (DL_EMP_code.txt):
Required a pre-trained gaze estimation model served via HTTP (local or Roboflow).
No model-serving code or Roboflow project was provided.
Relied on external API calls and base64-encoded frames.

## Final Offline Pipeline:

Uses open-source MediaPipe Face Mesh for direct landmark-based gaze computation.
Fully offline, no external dependencies beyond Python packages.
Transparent computations and reproducible results.

## License & Acknowledgements

This pilot uses Google’s MediaPipe framework under the Apache 2.0 license.
For a detailed explanation, code comments, and discussion of limitations and future work, please see GazeEstimatorReport.pdf.
- **Dr. Eungjoo Lee**, Assistant Professor, ECE @ University of Arizona — for the original eye-movement perimetry protocol and mentorship.
- **Paul Chong** — for sharing the initial workflow specification.

Prepared by Asifur Rahman, University of Arizona

Updated: Gaze Estimator Report – Version 2 added with stimulus correction and clinical metrics. 