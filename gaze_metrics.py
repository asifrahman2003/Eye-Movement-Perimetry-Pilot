"""
    @program: gaze_metrics.py
    @program description: Program performs extended analysis on gaze estimation data from a simulated visual field test. It calculates Saccadic Reaction Time (SRT) for yaw and pitch, matches actual gaze direction to expected direction using a predefined stimulus schedule, computes directional deviation and detection success, and generates visual field heatmaps. Also, outputs results to .csv and saves plots in 'analysis_outputs/' folder
"""

# required libraries
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

# loading peak matching file
peaks_df = pd.read_csv("gaze_peaks.csv")

# calculating Saccadic Reaction Time (SRT) for yaw and pitch
# rounding to 4 decimal place after the calculation to get more true difference first
peaks_df["srt_yaw"] = (peaks_df["yaw_peak_s"] - peaks_df["stim_time_s"]).round(4)
peaks_df["srt_pitch"] = (peaks_df["pitch_peak_s"] - peaks_df["stim_time_s"]).round(4)

# only keeping valid (not NaN) SRTs
valid_srt_df = peaks_df.dropna(subset=["srt_yaw", "srt_pitch"])

# saving to a new .csv file
os.makedirs("analysis_outputs", exist_ok=True)
valid_srt_df.to_csv("srt_results.csv", index=False)
print("SRT results saved as srt_results.csv")

# printing summary
print("\nSRT Summary:")
print(f"Yaw SRTs: {valid_srt_df['srt_yaw'].mean():.3f}s ± {valid_srt_df['srt_yaw'].std():.3f}s")
print(f"Pitch SRTs: {valid_srt_df['srt_pitch'].mean():.3f}s ± {valid_srt_df['srt_pitch'].std():.3f}s")

# now defining a list of predicatable stimulus positions (every 2s)
# since the video was 36 seconds in total (18 stimuli)
# hard-coding the dot position in the video for prototyping
stimulus_directions = [
    "down", "up", "down-right", "down-left", "right",
    "left", "up", "left", "up-left", 
    "right", "down", "up-right"
]

# Gaze deviation analysis
# loading raw gaze time series
gaze_df = pd.read_csv("mediapipe_gaze.csv")

# ideal stimulus times (3s intervals for 12 stimuli)
stim_times = np.arange(3, 3 + len(stimulus_directions) * 3, 3)

# collecting actual gaze for each stimulus
records = []

for stim_time, direction in zip(stim_times, stimulus_directions):
    window = gaze_df[(gaze_df["time_s"] >= stim_time + 0.3) & (gaze_df["time_s"] <= stim_time + 0.8)]
    if not window.empty:
        mean_yaw = window["yaw"].mean()
        mean_pitch = window["pitch"].mean()
        records.append({"stim_time_s": stim_time, "direction": direction, "actual_yaw": mean_yaw, "actual_pitch": mean_pitch})

# computing mean yaw/pitch per direction (signature)
record_df = pd.DataFrame.from_records(records)
signature_df = record_df.groupby("direction")[["actual_yaw", "actual_pitch"]].mean().rename(columns={"actual_yaw": "ideal_yaw", "actual_pitch": "ideal_pitch"})

# merging to get ideal values and computing error
merged_df = record_df.merge(signature_df, on="direction")
merged_df["yaw_error"] = merged_df["actual_yaw"] - merged_df["ideal_yaw"]
merged_df["pitch_error"] = merged_df["actual_pitch"] - merged_df["ideal_pitch"]
merged_df["total_deviation"] = np.sqrt(merged_df["yaw_error"] ** 2 + merged_df["pitch_error"] ** 2)

# direction-based detection (vs peak-based)
YAW_TOLERANCE = 0.1
PITCH_TOLERANCE = 0.1

merged_df["directionally_detected"] = (
    (merged_df["yaw_error"].abs() <= YAW_TOLERANCE) &
    (merged_df["pitch_error"].abs() <= PITCH_TOLERANCE)
)

# printing summary
n_detected = merged_df["directionally_detected"].sum()
total = len(merged_df)
print(f"\nDirectionally Matched Detections: {n_detected}/{total} ({n_detected/total*100:.1f}%)")

# saving the updated results
merged_df.to_csv("deviation_results.csv", index=False)
print("Deviation results saved as deviation_results.csv")

# summarizing deviation per direction
summary = merged_df.groupby("direction")["total_deviation"].mean().sort_values()
print("\nMean Deviation by Direction: ")
print(summary)
summary.to_csv("deviation_summary_by_direction.csv")

# visualizing plot for deviation summary (would be helpful for debugging in future, in case)
plt.figure(figsize=(10, 5))
summary.plot(kind="barh", color="skyblue")
plt.xlabel("Average Total Deviation")
plt.title("Mean Gaze Deviation by Stimulus Direction")
plt.tight_layout()
plt.savefig("analysis_outputs/gaze_deviation_summary.png")
plt.show()

### Visual Field Heatmap of Gaze Deviation
# direction to (x, y) coordinate mapping
direction_coords = {
    "up-left": (-1, 1), "up": (0, 1), "up-right": (1, 1),
    "left": (-1, 0), "center": (0, 0), "right": (1, 0),
    "down-left": (-1, -1), "down": (0, -1), "down-right": (1, -1)
}

# mapping each row in merged_df to its coordinate
heatmap_data = []
for _, row in merged_df.iterrows():
    if row["direction"] in direction_coords:
        x, y = direction_coords[row["direction"]]
        heatmap_data.append({"x": x, "y": y, "deviation": row["total_deviation"]})

heatmap_df = pd.DataFrame(heatmap_data)

# average the deviation per grid cell
heatmap_grid = heatmap_df.groupby(["y", "x"])["deviation"].mean().unstack()

# plotting the heatmap (inverting y-axis to match visual field convention)
plt.figure(figsize=(6, 6))
plt.imshow(heatmap_grid.sort_index(ascending=False), cmap="hot", interpolation="nearest")
plt.colorbar(label="Mean Deviation")
plt.xticks(ticks=range(3), labels=["Left", "Center", "Right"])
plt.yticks(ticks=range(3), labels=["Up", "Center", "Down"])
plt.title("Visual Field Heatmap: Gaze Deviation")
plt.tight_layout()
plt.savefig("analysis_outputs/visual_field_heatmap.png")
plt.show()


### visual field heatmap for detection Success
heatmap_detection = []
for _, row in merged_df.iterrows():
    if row["direction"] in direction_coords:
        x, y = direction_coords[row["direction"]]
        heatmap_detection.append({"x": x, "y": y, "detected": int(row["directionally_detected"])})

detect_df = pd.DataFrame(heatmap_detection)
detect_grid = detect_df.groupby(["y", "x"])["detected"].mean().unstack()

plt.figure(figsize=(6, 6))
plt.imshow(detect_grid.sort_index(ascending=False), cmap="Greens", interpolation="nearest", vmin=0, vmax=1)
plt.colorbar(label="Detection Rate")
plt.xticks(ticks=range(3), labels=["Left", "Center", "Right"])
plt.yticks(ticks=range(3), labels=["Up", "Center", "Down"])
plt.title("Visual Field Heatmap: Detection Success (Direction Match)")
plt.tight_layout()
plt.savefig("analysis_outputs/visual_field_detection_map.png")
plt.show()


### Heatmap of SRT values per direction
srt_heatmap_data = []
for _, row in peaks_df.iterrows():
    if row["stim_time_s"] in stim_times:
        idx = list(stim_times).index(row["stim_time_s"])
        direction = stimulus_directions[idx]
        if direction in direction_coords:
            x, y = direction_coords[direction]
            srt = row["srt_yaw"] if not pd.isna(row["srt_yaw"]) else np.nan
            srt_heatmap_data.append({"x": x, "y": y, "srt_yaw": srt})

srt_df = pd.DataFrame(srt_heatmap_data)
srt_grid = srt_df.groupby(["y", "x"])["srt_yaw"].mean().unstack()

plt.figure(figsize=(6, 6))
plt.imshow(srt_grid.sort_index(ascending=False), cmap="coolwarm", interpolation="nearest")
plt.colorbar(label="Mean Yaw SRT (s)")
plt.xticks(ticks=range(3), labels=["Left", "Center", "Right"])
plt.yticks(ticks=range(3), labels=["Up", "Center", "Down"])
plt.title("Visual Field Heatmap: Yaw Saccadic Reaction Time")
plt.tight_layout()
plt.savefig("analysis_outputs/visual_field_srt_heatmap.png")
plt.show()

# summary of missed SRTs
total_stimuli = len(peaks_df)
missed_yaw = peaks_df["srt_yaw"].isna().sum()
missed_pitch = peaks_df["srt_pitch"].isna().sum()
print(f"\nMissed Yaw SRTs: {missed_yaw}/{total_stimuli} ({missed_yaw/total_stimuli*100:.1f}%)")
print(f"Missed Pitch SRTs: {missed_pitch}/{total_stimuli} ({missed_pitch/total_stimuli*100:.1f}%)")

### tasks completion message
print("\nAll analysis complete. Resulting graphs saved to directory subfolder analysis_outputs/")