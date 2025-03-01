import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import time
from collections import deque

# Load MoveNet model from TensorFlow Hub
module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
movenet = module.signatures['serving_default']

# Full-body skeleton connections (including right arm)
KEYPOINT_EDGES = {
    (0, 1): (255, 0, 0),   # Nose → Left Eye
    (0, 2): (0, 0, 255),   # Nose → Right Eye
    (1, 3): (255, 0, 0),   # Left Eye → Left Ear
    (2, 4): (0, 0, 255),   # Right Eye → Right Ear
    (0, 5): (255, 255, 0), # Nose → Left Shoulder
    (0, 6): (0, 255, 255), # Nose → Right Shoulder
    (5, 7): (255, 255, 0), # Left Shoulder → Left Elbow
    (7, 9): (255, 255, 0), # Left Elbow → Left Wrist
    (6, 8): (0, 255, 255), # Right Shoulder → Right Elbow
    (8, 10): (0, 255, 255),# Right Elbow → Right Wrist
    (5, 6): (0, 255, 0),   # Left Shoulder → Right Shoulder
    (5, 11): (255, 255, 0),# Left Shoulder → Left Hip
    (6, 12): (0, 255, 255),# Right Shoulder → Right Hip
    (11, 12): (0, 255, 0), # Left Hip → Right Hip
    (11, 13): (255, 255, 0),# Left Hip → Left Knee
    (13, 15): (255, 255, 0),# Left Knee → Left Ankle
    (12, 14): (0, 255, 255),# Right Hip → Right Knee
    (14, 16): (0, 255, 255) # Right Knee → Right Ankle
}

# Labels for each keypoint
BODY_PARTS = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

# Right arm tracking for chop detection
RIGHT_ARM_PARTS = {
    "right_shoulder": 6, "right_elbow": 8, "right_wrist": 10
}

# Store right wrist motion history for chop detection
motion_history = deque(maxlen=10)  # Store last 10 positions
chop_count = 0
last_motion = None  # Stores last movement direction ("up" or "down")

# Variables for jump and squat detection using the hip center
baseline_hip_y = None         # Baseline hip center (to be calibrated)
calibration_frames = []       # List to collect initial hip center positions
in_jump = False               # Whether currently in an upward jump motion
in_squat = False              # Whether currently in a squat (downward motion)
jump_count = 0
squat_count = 0

def detect_pose(frame):
    """Run MoveNet on a single frame and return keypoints."""
    input_image = tf.image.resize_with_pad(np.expand_dims(frame, axis=0), 256, 256)
    input_image = tf.cast(input_image, dtype=tf.int32)
    outputs = movenet(input_image)
    keypoints = outputs['output_0'].numpy()
    return keypoints

def count_chops(keypoints):
    """Detect up-and-down chopping motion and count chops using right wrist."""
    global motion_history, chop_count, last_motion
    height, width = 480, 640  # Frame dimensions

    # Extract right wrist position
    right_wrist = keypoints[0, 0, RIGHT_ARM_PARTS["right_wrist"], :2]  # (y, x)
    right_wrist_y = right_wrist[0] * height  # Convert normalized value to pixels

    # Store right wrist position for motion tracking
    motion_history.append(right_wrist_y)

    # Detect chop motion based on recent movement
    if len(motion_history) > 5:
        prev_wrist_y = motion_history[-5]  # Compare with an earlier position
        if prev_wrist_y - right_wrist_y > 30:  # Moving down fast
            if last_motion == "up":  # If previous motion was up, count a chop
                chop_count += 1
                print(f"Chops: {chop_count}")
            last_motion = "down"
        elif right_wrist_y - prev_wrist_y > 30:  # Moving up fast
            last_motion = "up"

def count_jump_and_squat(keypoints, frame_height):
    """Detect jump and squat motions using the hip center.
    
    Jump is detected when the hip center moves upward fast relative to baseline,
    then returns to near baseline. Squat is detected when the hip center moves downward,
    then returns upward to baseline.
    """
    global baseline_hip_y, calibration_frames, in_jump, in_squat, jump_count, squat_count

    # Get left and right hip positions (indices 11 and 12)
    left_hip = keypoints[0, 0, 11, :2]
    right_hip = keypoints[0, 0, 12, :2]
    # Compute average hip y position (scaled to pixels)
    current_hip_y = ((left_hip[0] + right_hip[0]) / 2) * frame_height

    # Calibration: use first 30 frames to set the baseline hip position
    if baseline_hip_y is None:
        calibration_frames.append(current_hip_y)
        if len(calibration_frames) >= 30:
            baseline_hip_y = sum(calibration_frames) / len(calibration_frames)
            print(f"Baseline hip position established: {baseline_hip_y:.2f}")
        return  # Skip detection until calibration is complete

    # Thresholds (in pixels) – adjust these values as needed for your setup
    jump_threshold = 20   # Minimum upward movement (baseline - current_hip_y)
    squat_threshold = 20  # Minimum downward movement (current_hip_y - baseline)
    reset_threshold = 10  # When hip is within this range of baseline, consider motion complete

    # Jump detection (hip moves upward from baseline then returns)
    if not in_jump and (baseline_hip_y - current_hip_y > jump_threshold):
        in_jump = True
    elif in_jump and abs(current_hip_y - baseline_hip_y) < reset_threshold:
        jump_count += 1
        in_jump = False
        print(f"Jumps: {jump_count}")

    # Squat detection (hip moves downward from baseline then returns)
    if not in_squat and (current_hip_y - baseline_hip_y > squat_threshold):
        in_squat = True
    elif in_squat and abs(current_hip_y - baseline_hip_y) < reset_threshold:
        squat_count += 1
        in_squat = False
        print(f"Squats: {squat_count}")

def draw_skeleton_and_labels(frame, keypoints, threshold=0.3):
    """Draw the entire body skeleton and add labels."""
    height, width, _ = frame.shape
    points = {}

    for idx in range(17):  # Loop through all keypoints
        y, x, confidence = keypoints[0, 0, idx]
        if confidence > threshold:
            px, py = int(x * width), int(y * height)
            points[idx] = (px, py)
            cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)  # Draw keypoint
            cv2.putText(frame, BODY_PARTS[idx], (px + 5, py - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw skeleton lines
    for (p1, p2), color in KEYPOINT_EDGES.items():
        if p1 in points and p2 in points:
            cv2.line(frame, points[p1], points[p2], color, 2)

    # Optionally, display counts on the frame
    cv2.putText(frame, f"Chops: {chop_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Jumps: {jump_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Squats: {squat_count}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    return frame

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect pose
    keypoints = detect_pose(frame_rgb)

    # Detect chopping motion and count chops
    count_chops(keypoints)

    # Detect jump and squat motions using the hip center
    frame_height, frame_width = frame.shape[:2]
    count_jump_and_squat(keypoints, frame_height)

    # Draw full body skeleton with labels and display counts
    frame = draw_skeleton_and_labels(frame, keypoints)

    # Display frame
    cv2.imshow('Chopping Tree Game', frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
