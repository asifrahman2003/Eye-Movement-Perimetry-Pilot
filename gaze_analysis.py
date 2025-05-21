"""
    @program: gaze_analysis.py
    @description: loads gaze time series (yaw and pitch) from .csv, and smooths signals,
                  detects gaze peaks for stimulus analysis, computes metrics and generates
                  summary plots
"""

# required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
import os

# Loads the gaze .csv file
df = pd.read_csv("mediapipe_gaze.csv")  
os.makedirs("analysis_outputs", exist_ok=True)
times = df["time_s"].values
yaw = df["yaw"].interpolate().bfill().ffill().values
pitch = df["pitch"].interpolate().bfill().ffill().values

# Smooths the signals over a small window (in our case, 5 frames)
window_size = 5
yaw_s   = uniform_filter1d(yaw, size=window_size)
pitch_s = uniform_filter1d(pitch, size=window_size)

# Estimates FPS and define ideal stimulus times
fps = 1 / np.median(np.diff(times))
max_time = times[-1]
ideal_times = np.arange(2, max_time, 2)

# Detects peaks in yaw & pitch
min_distance = int(0.8 * fps)  # at least 0.8 s between peaks
yaw_peaks,   _ = find_peaks(yaw_s,   distance=min_distance)
pitch_peaks, _ = find_peaks(pitch_s, distance=min_distance)

yaw_times   = times[yaw_peaks]
pitch_times = times[pitch_peaks]

# Matches detected peaks to ideal stimuli
tolerance = 0.5  # in seconds

records = []
for stim in ideal_times:
    # yaw
    if len(yaw_times)>0:
        iy = np.argmin(np.abs(yaw_times - stim))
        dy = yaw_times[iy] - stim
        yaw_match = yaw_times[iy] if abs(dy)<=tolerance else np.nan
        err_yaw   = dy if abs(dy)<=tolerance else np.nan
    else:
        yaw_match = err_yaw = np.nan
    # pitch
    if len(pitch_times)>0:
        ip = np.argmin(np.abs(pitch_times - stim))
        dp = pitch_times[ip] - stim
        pitch_match = pitch_times[ip] if abs(dp)<=tolerance else np.nan
        err_pitch   = dp if abs(dp)<=tolerance else np.nan
    else:
        pitch_match = err_pitch = np.nan

    records.append({
        "stim_time_s": float(stim),
        "yaw_peak_s": yaw_match,
        "yaw_err_s": err_yaw,
        "pitch_peak_s": pitch_match,
        "pitch_err_s": err_pitch
    })

peaks_df = pd.DataFrame.from_records(records)

# Computes the metrics
def summarize(err_col):
    valid = peaks_df[err_col].dropna().abs()
    caught = len(valid)
    total  = len(peaks_df)
    return caught, total, caught/total*100, valid.mean()

yaw_c, total, yaw_acc, yaw_me = summarize("yaw_err_s")
pitch_c, _, pitch_acc, pitch_me = summarize("pitch_err_s")

print(f"Estimated FPS: {fps:.2f}")
print(f"Yaw   → {yaw_c}/{total} caught  ({yaw_acc:.1f}%),  mean err {yaw_me:.3f}s")
print(f"Pitch → {pitch_c}/{total} caught  ({pitch_acc:.1f}%),  mean err {pitch_me:.3f}s")

# 7) Saves the summary to another .csv file
peaks_df.to_csv("gaze_peaks.csv", index=False)
print("Saved detailed peak matches to gaze_peaks.csv")

# 8) Plotting the graphs for analysis
for signal, peaks, label in [(yaw_s, yaw_peaks, "Yaw"), (pitch_s, pitch_peaks, "Pitch")]:
    plt.figure(figsize=(10,4))
    plt.plot(times, signal, label=label + " (smoothed)")
    plt.plot(times[peaks], signal[peaks], marker="x", linestyle="None", label="Detected Peaks")
    for t in ideal_times:
        plt.axvline(t, linestyle="--", linewidth=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel(label)
    plt.title(f"{label} vs Time with Stimuli & Detected Peaks")
    plt.legend()
    plt.tight_layout()
    # Saves the figure
    out_path = os.path.join("analysis_outputs", f"{label.lower()}_plot.png")
    plt.savefig(out_path)
    print(f"Saved {label} plot to {out_path}")
    plt.show()