import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_docs.vis import embed
import numpy as np
import cv2

# Import matplotlib libraries
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches

# Some modules to display an animation using imageio.
import imageio
from IPython.display import HTML, display

def load_movenet_model(model_name="movenet_lightning"):
    """Loads the MoveNet model and returns the model function and input size."""
    if "tflite" in model_name:
        import subprocess
        
        if "movenet_lightning_f16" in model_name:
            subprocess.run(["wget", "-q", "-O", "model.tflite", 
                           "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite"])
            input_size = 192
        elif "movenet_thunder_f16" in model_name:
            subprocess.run(["wget", "-q", "-O", "model.tflite", 
                           "https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4?lite-format=tflite"])
            input_size = 256
        elif "movenet_lightning_int8" in model_name:
            subprocess.run(["wget", "-q", "-O", "model.tflite", 
                           "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4?lite-format=tflite"])
            input_size = 192
        elif "movenet_thunder_int8" in model_name:
            subprocess.run(["wget", "-q", "-O", "model.tflite", 
                           "https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/int8/4?lite-format=tflite"])
            input_size = 256
        else:
            raise ValueError("Unsupported model name: %s" % model_name)

        # Initialize the TFLite interpreter
        interpreter = tf.lite.Interpreter(model_path="model.tflite")
        interpreter.allocate_tensors()

        def movenet(input_image):
            """Runs detection on an input image.

            Args:
              input_image: A [1, height, width, 3] tensor represents the input image
                pixels. Note that the height/width should already be resized and match the
                expected input resolution of the model before passing into this function.

            Returns:
              A [1, 1, 17, 3] float numpy array representing the predicted keypoint
              coordinates and scores.
            """
            # TF Lite format expects tensor type of uint8.
            input_image = tf.cast(input_image, dtype=tf.uint8)
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
            # Invoke inference.
            interpreter.invoke()
            # Get the model prediction.
            keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
            return keypoints_with_scores

    else:
        if "movenet_lightning" in model_name:
            module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
            input_size = 192
        elif "movenet_thunder" in model_name:
            module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
            input_size = 256
        else:
            raise ValueError("Unsupported model name: %s" % model_name)

        def movenet(input_image):
            """Runs detection on an input image.

            Args:
              input_image: A [1, height, width, 3] tensor represents the input image
                pixels. Note that the height/width should already be resized and match the
                expected input resolution of the model before passing into this function.

            Returns:
              A [1, 1, 17, 3] float numpy array representing the predicted keypoint
              coordinates and scores.
            """
            model = module.signatures['serving_default']

            # SavedModel format expects tensor type of int32.
            input_image = tf.cast(input_image, dtype=tf.int32)
            # Run model inference.
            outputs = model(input_image)
            # Output is a [1, 1, 17, 3] tensor.
            keypoints_with_scores = outputs['output_0'].numpy()
            return keypoints_with_scores
    
    return movenet, input_size

def draw_prediction_on_image(
    image, keypoints_with_scores, crop_region=None, close_figure=False,
    output_image_height=None):
    """Draws the keypoint predictions on image."""
    height, width, channel = image.shape
    aspect_ratio = float(width) / height
    fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
    # To remove the huge white borders
    fig.tight_layout(pad=0)
    ax.margins(0)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.axis('off')

    ax.imshow(image)
    
    # Define keypoint connections and colors
    KEYPOINT_EDGE_INDS_TO_COLOR = {
        (0, 1): 'm',    # nose to left_eye
        (0, 2): 'c',    # nose to right_eye
        (1, 3): 'm',    # left_eye to left_ear
        (2, 4): 'c',    # right_eye to right_ear
        (0, 5): 'm',    # nose to left_shoulder
        (0, 6): 'c',    # nose to right_shoulder
        (5, 7): 'm',    # left_shoulder to left_elbow
        (7, 9): 'm',    # left_elbow to left_wrist
        (6, 8): 'c',    # right_shoulder to right_elbow
        (8, 10): 'c',   # right_elbow to right_wrist
        (5, 6): 'y',    # left_shoulder to right_shoulder
        (5, 11): 'm',   # left_shoulder to left_hip
        (6, 12): 'c',   # right_shoulder to right_hip
        (11, 12): 'y',  # left_hip to right_hip
        (11, 13): 'm',  # left_hip to left_knee
        (13, 15): 'm',  # left_knee to left_ankle
        (12, 14): 'c',  # right_hip to right_knee
        (14, 16): 'c'   # right_knee to right_ankle
    }
    
    keypoints = keypoints_with_scores[0, 0, :, :2]
    scores = keypoints_with_scores[0, 0, :, 2]
    
    # Draw the keypoints
    for i, (keypoint, score) in enumerate(zip(keypoints, scores)):
        if score >= 0.2:  # Adjust this threshold as needed
            y, x = keypoint
            ax.scatter(x * width, y * height, c='r', marker='o', s=50)
    
    # Draw the connections
    for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
        if (scores[edge_pair[0]] >= 0.2 and 
            scores[edge_pair[1]] >= 0.2):  # Adjust this threshold as needed
            
            y1, x1 = keypoints[edge_pair[0]]
            y2, x2 = keypoints[edge_pair[1]]
            ax.plot([x1 * width, x2 * width],
                   [y1 * height, y2 * height],
                   color=color, linewidth=2)
    
    # Draw the crop region if provided
    if crop_region is not None:
        ymin, xmin, ymax, xmax = crop_region
        rect = patches.Rectangle((xmin * width, ymin * height),
                                (xmax - xmin) * width, (ymax - ymin) * height,
                                linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(rect)
    
    fig.canvas.draw()
    
    # Fixed approach to convert the figure to a numpy array
    img = np.array(fig.canvas.renderer.buffer_rgba())
    
    # Convert RGBA to RGB
    img = img[:, :, :3]
    
    if output_image_height is not None:
        output_image_width = int(output_image_height * aspect_ratio)
        img = cv2.resize(img, (output_image_width, output_image_height))
    
    if close_figure:
        plt.close(fig)
    
    return img

def process_image(image_path, model_name="movenet_lightning"):
    """
    Process an image with MoveNet pose detection model.
    
    Args:
        image_path: Path to the input image
        model_name: Name of the MoveNet model to use
        
    Returns:
        None (saves the result image to disk)
    """
    # Load the MoveNet model
    movenet_model, input_size = load_movenet_model(model_name)
    print(f"Loaded {model_name} model with input size {input_size}")
    
    # Load the input image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)

    # Resize and pad the image to keep the aspect ratio and fit the expected size
    input_image = tf.expand_dims(image, axis=0)
    input_image = tf.image.resize_with_pad(input_image, input_size, input_size)

    # Run model inference
    keypoints_with_scores = movenet_model(input_image)
    
    # Print keypoints to see the results
    print("Keypoints with scores:")
    for i, score in enumerate(keypoints_with_scores[0, 0, :, 2]):
        if score > 0.2:
            print(f"Keypoint {i}: score = {score:.2f}")

    # Visualize the predictions with image
    display_image = tf.expand_dims(image, axis=0)
    display_image = tf.cast(tf.image.resize_with_pad(display_image, 1280, 1280), dtype=tf.int32)
    
    output_overlay = draw_prediction_on_image(np.squeeze(display_image.numpy(), axis=0), keypoints_with_scores)

    # Save the output image
    plt.figure(figsize=(5, 5))
    plt.imshow(output_overlay)
    plt.axis('off')
    plt.savefig('pose_detection_result.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print("Detection complete! Result saved as 'pose_detection_result.png'")

    """Runs model inference on the cropped region.

    Args:
      movenet_model: The MoveNet model function.
      image: A TF.Tensor with shape [height, width, 3] representing the image.
      crop_region: A dictionary that defines the crop region.
      crop_size: A list of two integers [height, width] specifying the size to crop.

    Returns:
      A numpy array with shape [1, 1, 17, 3] representing the predicted keypoints.
    """
    image_height, image_width, _ = image.shape
    
    # Crop the image
    y_min, x_min, y_max, x_max = (
        crop_region['y_min'], crop_region['x_min'], 
        crop_region['y_max'], crop_region['x_max']
    )
    
    # Ensure crop coordinates are within image boundaries
    y_min = max(0, y_min)
    x_min = max(0, x_min)
    y_max = min(image_height, y_max)
    x_max = min(image_width, x_max)
    
    # Crop and resize the image
    cropped = image[y_min:y_max, x_min:x_max, :]
    cropped_shape = cropped.shape
    
    if cropped_shape[0] == 0 or cropped_shape[1] == 0:
        # Invalid crop region, use the entire image
        cropped = image
    
    resized = tf.image.resize_with_pad(
        tf.expand_dims(cropped, axis=0), crop_size[0], crop_size[1])
    
    # Run model inference
    keypoints_with_scores = movenet_model(resized)
    
    # Adjust keypoint coordinates to the original image
    for idx in range(17):
        keypoints_with_scores[0, 0, idx, 0] = (
            crop_region['y_min'] + crop_region['height'] * 
            keypoints_with_scores[0, 0, idx, 0]) / image_height
        keypoints_with_scores[0, 0, idx, 1] = (
            crop_region['x_min'] + crop_region['width'] * 
            keypoints_with_scores[0, 0, idx, 1]) / image_width
    
    return keypoints_with_scores
    

def process_video(video_path, model_name="movenet_lightning", output_path="pose_detection_result.gif"):
    """
    Process a video with MoveNet pose detection model by slicing it into frames,
    processing each frame individually, and reassembling into a video.
    
    Args:
        video_path: Path to the input video/gif
        model_name: Name of the MoveNet model to use
        output_path: Path to save the output gif
        
    Returns:
        None (saves the result as a gif)
    """
    # Load the MoveNet model
    movenet_model, input_size = load_movenet_model(model_name)
    print(f"Loaded {model_name} model with input size {input_size}")
    
    # Load the input video using OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file {video_path}")
    
    # Get video dimensions and frame count
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing video with {frame_count} frames")
    
    # Process each frame
    output_images = []
    frame_idx = 0
    
    while cap.isOpened():
        # Read the next frame
        success, frame = cap.read()
        if not success:
            break
        
        # Convert BGR to RGB (OpenCV uses BGR by default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor
        frame_tensor = tf.convert_to_tensor(frame_rgb)
        
        # Resize and pad the image to keep the aspect ratio and fit the expected size
        input_image = tf.expand_dims(frame_tensor, axis=0)
        input_image = tf.image.resize_with_pad(input_image, input_size, input_size)

        # Run model inference
        keypoints_with_scores = movenet_model(input_image)
        
        # Visualize the predictions with image
        display_image = tf.expand_dims(frame_tensor, axis=0)
        display_image = tf.cast(tf.image.resize_with_pad(display_image, 1280, 1280), dtype=tf.int32)
        
        output_overlay = draw_prediction_on_image(
            np.squeeze(display_image.numpy(), axis=0), 
            keypoints_with_scores,
            close_figure=True,
            output_image_height=300)
        
        output_images.append(output_overlay)
        
        # Print progress
        if frame_idx % 10 == 0:
            print(f"Processed {frame_idx}/{frame_count} frames")
            
        frame_idx += 1
    
    # Release the video capture
    cap.release()
    
    # Create and save the output gif
    print("Creating output gif...")
    imageio.mimsave(output_path, output_images, duration=100/1000.0)
    
    print(f"Detection complete! Result saved as '{output_path}'")


if __name__ == "__main__":
    # For video processing
    process_video('tik.mp4')