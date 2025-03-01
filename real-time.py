import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import time
from collections import deque
import pyautogui

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

# Enhanced motion history tracking without maxlen limits
motion_history = {
    "right_wrist": [],     # Store all positions
    "right_elbow": [],
    "right_shoulder": [],
    "left_wrist": [],
    "left_elbow": [],
    "left_shoulder": [],
    "hip_center": [],      # For jump/squat tracking
    "timestamps": [],      # To track timing
    "actions": []          # Record detected actions with timestamps
}

chop_count = 0
jump_count = 0

# Game state variables
game_running = True

# Remove squat_cooldown and is_squatting from global variables
global chop_cooldown, jump_cooldown
chop_cooldown = 0
jump_cooldown = 0

# Add this constant near the other globals at the top of the file
JUMP_SENSITIVITY = 0.05  # Adjust this value to change jump detection sensitivity (lower = more sensitive)

# Add these near the other globals
WAVE_DETECTION_THRESHOLD = 15  # Pixel threshold for detecting hand wave
SQUAT_SENSITIVITY = 0.07  # Adjust for sensitivity (lower = more sensitive)
left_move_count = 0
right_move_count = 0
down_move_count = 0
left_move_cooldown = 0
right_move_cooldown = 0
down_move_cooldown = 0
is_squatting = False

def detect_pose(frame):
    """Run MoveNet on a single frame and return keypoints."""
    input_image = tf.image.resize_with_pad(np.expand_dims(frame, axis=0), 256, 256)
    input_image = tf.cast(input_image, dtype=tf.int32)
    outputs = movenet(input_image)
    keypoints = outputs['output_0'].numpy()
    return keypoints

def count_chops(keypoints):
    """Detect up-and-down chopping motion and count chops using right wrist."""
    global motion_history, chop_count
    height, width = 480, 640  # Frame dimensions

    # Extract right wrist position
    right_wrist = keypoints[0, 0, RIGHT_ARM_PARTS["right_wrist"], :2]  # (y, x)
    right_wrist_y = right_wrist[0] * height  # Convert normalized value to pixels

    # Store right wrist position for motion tracking
    motion_history["right_wrist"].append(right_wrist_y)

    # Detect chop motion based on recent movement
    if len(motion_history["right_wrist"]) > 5:
        prev_wrist_y = motion_history["right_wrist"][-5]  # Compare with an earlier position
        if prev_wrist_y - right_wrist_y > 30:  # Moving down fast
            chop_count += 1
            action_time = time.time()
            motion_history["actions"].append(f"{action_time:.2f}: Chop detected (#{chop_count})")
            print(f"Chops: {chop_count}")
            
            # Simulate a keyboard press for jump using pyautogui
            pyautogui.press('space')
            print("Jump triggered!")

def detect_jump(keypoints, frame_height):
    global jump_count, jump_cooldown
    
    # Get average shoulder y-position (shoulders go up when jumping)
    left_shoulder_y = keypoints[0, 0, 5, 0] * frame_height  # Left shoulder
    right_shoulder_y = keypoints[0, 0, 6, 0] * frame_height  # Right shoulder
    
    # Calculate average shoulder position
    shoulder_y = (left_shoulder_y + right_shoulder_y) / 2 if left_shoulder_y > 0 and right_shoulder_y > 0 else None
    
    # Add to motion history
    if shoulder_y is not None:
        motion_history["shoulder_y"] = motion_history.get("shoulder_y", []) + [shoulder_y]
    
    # Process jump detection
    current_time = time.time()
    if jump_cooldown < current_time:
        # Get recent shoulder positions for analysis
        recent_shoulder_positions = deque(maxlen=10)
        for pos in motion_history.get("shoulder_y", []):
            if pos is not None:
                recent_shoulder_positions.append(pos)
        
        if len(recent_shoulder_positions) >= 5:  # Need enough data points
            # Calculate average baseline position
            baseline = sum(list(recent_shoulder_positions)[-5:]) / 5
            
            # Check if current position is significantly higher than baseline
            # Lower shoulder y value means shoulders moved up in the frame
            if shoulder_y is not None and shoulder_y < baseline - (JUMP_SENSITIVITY * frame_height):
                jump_count += 1
                motion_history["actions"].append(f"{current_time}: Jump detected")
                jump_cooldown = current_time + 1.0  # Cooldown period
                # Trigger virtual key press for jump
                pyautogui.press('up')
                print(f"Jump detected! Count: {jump_count}")

def update_motion_history(keypoints, frame_height, frame_width):
    """Update comprehensive motion history with all key body parts."""
    # Record positions of key body parts
    # Left arm
    left_shoulder = keypoints[0, 0, 5, :2]
    left_elbow = keypoints[0, 0, 7, :2]
    left_wrist = keypoints[0, 0, 9, :2]
    
    # Right arm
    right_shoulder = keypoints[0, 0, 6, :2]
    right_elbow = keypoints[0, 0, 8, :2]
    right_wrist = keypoints[0, 0, 10, :2]
    
    # Store positions (convert from normalized to pixel coordinates)
    motion_history["left_shoulder"].append((left_shoulder[1] * frame_width, left_shoulder[0] * frame_height))
    motion_history["left_elbow"].append((left_elbow[1] * frame_width, left_elbow[0] * frame_height))
    motion_history["left_wrist"].append((left_wrist[1] * frame_width, left_wrist[0] * frame_height))
    
    motion_history["right_shoulder"].append((right_shoulder[1] * frame_width, right_shoulder[0] * frame_height))
    motion_history["right_elbow"].append((right_elbow[1] * frame_width, right_elbow[0] * frame_height))
    motion_history["right_wrist"].append((right_wrist[1] * frame_width, right_wrist[0] * frame_height))

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

    # Update the game information overlay to include all movement counts
    cv2.putText(frame, f"Chops: {chop_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Jumps: {jump_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Left: {left_move_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Right: {right_move_count}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Down: {down_move_count}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return frame

def detect_right_movement(keypoints, frame_width, frame_height):
    """Detect when user moves to their right (appears as leftward movement in camera)."""
    global right_move_count, right_move_cooldown
    
    # Get shoulder positions
    left_shoulder_x = keypoints[0, 0, 5, 1] * frame_width   # Left shoulder, x coordinate
    right_shoulder_x = keypoints[0, 0, 6, 1] * frame_width  # Right shoulder, x coordinate
    
    # Calculate shoulder center position
    shoulder_center_x = (left_shoulder_x + right_shoulder_x) / 2 if left_shoulder_x > 0 and right_shoulder_x > 0 else None
    
    # Store shoulder center position with timestamp
    current_time = time.time()
    if shoulder_center_x is not None:
        if "shoulder_center_x" not in motion_history:
            motion_history["shoulder_center_x"] = []
        motion_history["shoulder_center_x"].append((shoulder_center_x, current_time))
    
    # Only proceed if we have enough data and not in cooldown
    if "shoulder_center_x" in motion_history and len(motion_history["shoulder_center_x"]) >= 10:
        # Get recent positions (within last 1.5 seconds)
        recent_positions = [pos for pos in motion_history["shoulder_center_x"][-15:] 
                           if current_time - pos[1] < 1.5]
        
        if len(recent_positions) >= 5:
            # Calculate baseline from first few frames
            baseline_x = sum([pos[0] for pos in recent_positions[:5]]) / 5
            
            # Current position (average of last few frames to reduce noise)
            current_x = sum([pos[0] for pos in recent_positions[-3:]]) / 3
            
            # Define movement threshold (how far shoulders need to move to trigger)
            movement_threshold = frame_width * 0.06  # 6% of frame width
            
            # When user moves to their right, they appear to move left in camera
            # So we check for decreasing x value
            if right_move_cooldown < current_time and current_x < baseline_x - movement_threshold:
                right_move_count += 1
                motion_history["actions"].append(f"{current_time:.2f}: Right movement detected (#{right_move_count})")
                right_move_cooldown = current_time + 0.8  # 0.8 second cooldown
                pyautogui.press('right')  # Press right key when user moves right
                print(f"Right movement detected! Count: {right_move_count}")

def detect_left_movement(keypoints, frame_width, frame_height):
    """Detect when user moves to their left (appears as rightward movement in camera)."""
    global left_move_count, left_move_cooldown
    
    # Get shoulder positions
    left_shoulder_x = keypoints[0, 0, 5, 1] * frame_width   # Left shoulder, x coordinate
    right_shoulder_x = keypoints[0, 0, 6, 1] * frame_width  # Right shoulder, x coordinate
    
    # Calculate shoulder center position
    shoulder_center_x = (left_shoulder_x + right_shoulder_x) / 2 if left_shoulder_x > 0 and right_shoulder_x > 0 else None
    
    # Store shoulder center position with timestamp
    current_time = time.time()
    if shoulder_center_x is not None:
        if "shoulder_center_x" not in motion_history:
            motion_history["shoulder_center_x"] = []
        motion_history["shoulder_center_x"].append((shoulder_center_x, current_time))
    
    # Only proceed if we have enough data and not in cooldown
    if "shoulder_center_x" in motion_history and len(motion_history["shoulder_center_x"]) >= 10:
        # Get recent positions (within last 1.5 seconds)
        recent_positions = [pos for pos in motion_history["shoulder_center_x"][-15:] 
                           if current_time - pos[1] < 1.5]
        
        if len(recent_positions) >= 5:
            # Calculate baseline from first few frames
            baseline_x = sum([pos[0] for pos in recent_positions[:5]]) / 5
            
            # Current position (average of last few frames to reduce noise)
            current_x = sum([pos[0] for pos in recent_positions[-3:]]) / 3
            
            # Define movement threshold (how far shoulders need to move to trigger)
            movement_threshold = frame_width * 0.06  # 6% of frame width
            
            # When user moves to their left, they appear to move right in camera
            # So we check for increasing x value
            if left_move_cooldown < current_time and current_x > baseline_x + movement_threshold:
                left_move_count += 1
                motion_history["actions"].append(f"{current_time:.2f}: Left movement detected (#{left_move_count})")
                left_move_cooldown = current_time + 0.8  # 0.8 second cooldown
                pyautogui.press('left')  # Press left key when user moves left
                print(f"Left movement detected! Count: {left_move_count}")

def detect_squat(keypoints, frame_height):
    """Detect squatting motion for downward movement."""
    global down_move_count, down_move_cooldown, is_squatting
    
    # Get hip positions
    left_hip_y = keypoints[0, 0, 11, 0] * frame_height   # Left hip
    right_hip_y = keypoints[0, 0, 12, 0] * frame_height  # Right hip
    
    # Calculate hip center position
    hip_center_y = (left_hip_y + right_hip_y) / 2 if left_hip_y > 0 and right_hip_y > 0 else None
    
    # Store hip center position
    if hip_center_y is not None:
        if "hip_center_y" not in motion_history:
            motion_history["hip_center_y"] = []
        motion_history["hip_center_y"].append(hip_center_y)
    
    # Process squat detection
    current_time = time.time()
    if down_move_cooldown < current_time and hip_center_y is not None:
        if "hip_center_y" in motion_history and len(motion_history["hip_center_y"]) >= 10:
            # Calculate baseline (standing position) from earlier frames
            baseline = sum(motion_history["hip_center_y"][-10:-5]) / 5
            
            # Current position (average of last few frames to reduce noise)
            current_pos = sum(motion_history["hip_center_y"][-3:]) / 3
            
            # Detect squatting (hip position lower than baseline)
            if not is_squatting and current_pos > baseline + (SQUAT_SENSITIVITY * frame_height):
                is_squatting = True
                down_move_count += 1
                motion_history["actions"].append(f"{current_time:.2f}: Squat detected (#{down_move_count})")
                down_move_cooldown = current_time + 1.0  # 1 second cooldown
                pyautogui.press('down')
                print(f"Squat detected! Count: {down_move_count}")
            
            # Reset squatting state when user stands back up
            elif is_squatting and current_pos < baseline + (SQUAT_SENSITIVITY * frame_height * 0.5):
                is_squatting = False

# Open webcam - Try multiple camera indices if needed
camera_index = 0
max_attempts = 3

while camera_index < max_attempts:
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        # Check if this is the right camera by grabbing a test frame
        ret, test_frame = cap.read()
        if ret:
            print(f"Successfully opened camera at index {camera_index}")
            break
    
    print(f"Failed to open camera at index {camera_index}")
    camera_index += 1
    
if not cap.isOpened():
    print("Error: Could not open any webcam.")
    exit()

# Create window
cv2.namedWindow('Chopping Tree Game')

while cap.isOpened() and game_running:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect pose
    keypoints = detect_pose(frame_rgb)

    # Update comprehensive motion history
    frame_height, frame_width = frame.shape[:2]
    update_motion_history(keypoints, frame_height, frame_width)

    # Detect chopping motion and count chops
    count_chops(keypoints)

    # Detect jump motions using the hip center
    detect_jump(keypoints, frame_height)

    # Separate left and right movement detection
    detect_left_movement(keypoints, frame_width, frame_height)
    detect_right_movement(keypoints, frame_width, frame_height)
    detect_squat(keypoints, frame_height)

    # Draw full body skeleton with labels and display counts
    frame = draw_skeleton_and_labels(frame, keypoints)

    # Display frame
    cv2.imshow('Chopping Tree Game', frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Print motion history after game ends

for action in motion_history["actions"]:
    print(action)
    print(f"\n{part.replace('_', ' ').title()} positions:")   
    if len(motion_history[part]) > 10:
        # Print first 5 and last 5 entries
        for i in range(5):
            print(f"Frame {i}: {motion_history[part][i]}")
        print("...")
        for i in range(5):
            idx = len(motion_history[part]) - 5 + i
            print(f"Frame {idx}: {motion_history[part][idx]}")
    else:
        # Print all entries if less than 10
        for i, pos in enumerate(motion_history[part]):
            print(f"Frame {i}: {pos}")

# Update final summary to include all movement counts
print("\n--- Game Summary ---")
print(f"Final Chop Count: {chop_count}")
print(f"Final Jump Count: {jump_count}")
print(f"Final Left Movement Count: {left_move_count}")
print(f"Final Right Movement Count: {right_move_count}")
print(f"Final Squat (Down) Count: {down_move_count}")

cap.release()
cv2.destroyAllWindows()
