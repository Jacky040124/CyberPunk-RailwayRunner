import cv2
import pygame
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from collections import deque

# --- Setup for Pose Detection using MoveNet ---
module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
movenet = module.signatures['serving_default']

# Global variables for chop detection using right wrist
motion_history = deque(maxlen=10)
chop_count = 0
last_motion = None  # "up" or "down"

# --- Pygame Game Implementation ---
def main():
    global chop_count, last_motion, motion_history

    pygame.init()
    screen_width, screen_height = 640, 480
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Tree Chopper Game")
    clock = pygame.time.Clock()

    # Define camera feed size (for display within the game window)
    cam_width, cam_height = 200, 150

    # Open the webcam via OpenCV
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    # Game state variables
    game_state = "start"  # "start", "playing", "game_over"
    start_time = None
    game_duration = 60 * 1000  # 60 seconds in milliseconds
    tree_count = 0

    # Tree falling animation control
    tree_falling = False
    falling_start_time = None
    falling_duration = 1000  # falling animation duration in milliseconds

    # UI fonts
    large_font = pygame.font.SysFont("Arial", 48)
    medium_font = pygame.font.SysFont("Arial", 36)
    small_font = pygame.font.SysFont("Arial", 24)

    running = True
    while running:
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if game_state == "start":
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    game_state = "playing"
                    start_time = pygame.time.get_ticks()
                    # Reset detection variables for fresh game
                    chop_count = 0
                    last_motion = None
                    motion_history.clear()
            if game_state == "game_over":
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    game_state = "playing"
                    start_time = pygame.time.get_ticks()
                    chop_count = 0
                    tree_count = 0
                    last_motion = None
                    motion_history.clear()
                    tree_falling = False
                    falling_start_time = None

        # Fill background with a dark gray color
        screen.fill((30, 30, 30))

        if game_state == "start":
            start_text = large_font.render("Tree Chopper Challenge", True, (255, 255, 255))
            instr_text = medium_font.render("Press SPACE to start", True, (200, 200, 200))
            screen.blit(start_text, (screen_width // 2 - start_text.get_width() // 2, 150))
            screen.blit(instr_text, (screen_width // 2 - instr_text.get_width() // 2, 220))

        elif game_state == "playing":
            # Update timer
            elapsed = pygame.time.get_ticks() - start_time
            remaining = max(0, game_duration - elapsed)
            if remaining == 0:
                game_state = "game_over"

            # --- Pose Detection ---
            ret, frame = cap.read()
            if ret:
                # Convert frame from BGR to RGB and process with MoveNet
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_image = tf.image.resize_with_pad(np.expand_dims(frame_rgb, axis=0), 256, 256)
                input_image = tf.cast(input_image, dtype=tf.int32)
                outputs = movenet(input_image)
                keypoints = outputs['output_0'].numpy()

                # Extract right wrist keypoint (index 10)
                height = screen_height  # reference height for game
                right_wrist = keypoints[0, 0, 10, :2]  # (y, x) normalized
                right_wrist_y = right_wrist[0] * height

                # Update motion history for chop detection
                motion_history.append(right_wrist_y)
                if len(motion_history) > 5:
                    prev_wrist_y = motion_history[-5]
                    if prev_wrist_y - right_wrist_y > 30:  # downward fast -> chop
                        if last_motion == "up":
                            chop_count += 1
                        last_motion = "down"
                    elif right_wrist_y - prev_wrist_y > 30:  # upward fast
                        last_motion = "up"

                # Create a Pygame surface for the camera feed
                cam_surface = pygame.surfarray.make_surface(np.rot90(frame_rgb))
                cam_surface = pygame.transform.scale(cam_surface, (cam_width, cam_height))
                # Draw a border around the camera feed
                cam_rect = pygame.Rect(screen_width - cam_width - 10, 10, cam_width, cam_height)
                pygame.draw.rect(screen, (255, 255, 255), cam_rect, 2)
                screen.blit(cam_surface, (screen_width - cam_width - 10, 10))
                # Label the camera feed
                cam_label = small_font.render("Camera Feed", True, (255, 255, 255))
                screen.blit(cam_label, (screen_width - cam_width - 10, cam_rect.bottom + 5))

            # --- Tree Logic & Health Animation ---
            # If 10 chops are reached and tree is not already falling, trigger fall animation
            if chop_count >= 10 and not tree_falling:
                tree_falling = True
                falling_start_time = pygame.time.get_ticks()

            # Define tree parameters (centered on left half of the screen)
            tree_x = screen_width // 2 - 100
            tree_y = screen_height // 2 + 30
            tree_width = 50
            tree_height = 150

            # Compute tree health (from 1.0 down to 0.0)
            health_ratio = max(0, 1.0 - (chop_count / 10.0))
            # Interpolate color: healthy green to damaged brown
            healthy_color = np.array([34, 139, 34])
            damaged_color = np.array([139, 69, 19])
            tree_color = healthy_color * health_ratio + damaged_color * (1 - health_ratio)
            tree_color = tuple(tree_color.astype(int))

            if tree_falling:
                progress = (pygame.time.get_ticks() - falling_start_time) / falling_duration
                # Create tree surface and rotate for falling animation
                tree_surface = pygame.Surface((tree_width, tree_height), pygame.SRCALPHA)
                tree_surface.fill(tree_color)
                angle = progress * 90  # up to 90 degrees rotation
                rotated_tree = pygame.transform.rotate(tree_surface, angle)
                rect = rotated_tree.get_rect(center=(tree_x, tree_y))
                screen.blit(rotated_tree, rect)
                if progress >= 1.0:
                    tree_count += 1
                    chop_count = 0
                    tree_falling = False
                    last_motion = None
                    motion_history.clear()
            else:
                # Draw standing tree
                pygame.draw.rect(screen, tree_color,
                                 (tree_x - tree_width // 2, tree_y - tree_height // 2,
                                  tree_width, tree_height))

            # Draw a health bar above the tree
            bar_width = 60
            bar_height = 10
            bar_x = tree_x - bar_width // 2
            bar_y = tree_y - tree_height // 2 - 20
            pygame.draw.rect(screen, (255, 255, 255), (bar_x, bar_y, bar_width, bar_height), 2)
            fill_width = int(bar_width * health_ratio)
            pygame.draw.rect(screen, (0, 255, 0), (bar_x, bar_y, fill_width, bar_height))

            # --- Draw UI Elements ---
            timer_text = medium_font.render(f"Time: {remaining // 1000}", True, (255, 255, 255))
            chop_text = medium_font.render(f"Chops: {chop_count}", True, (255, 255, 255))
            tree_text = medium_font.render(f"Trees: {tree_count}", True, (255, 255, 255))
            screen.blit(timer_text, (20, 20))
            screen.blit(chop_text, (20, 60))
            screen.blit(tree_text, (20, 100))

        elif game_state == "game_over":
            over_text = large_font.render(f"Game Over!", True, (255, 255, 255))
            score_text = medium_font.render(f"Trees chopped: {tree_count}", True, (255, 255, 255))
            restart_text = medium_font.render("Press R to Restart", True, (200, 200, 200))
            screen.blit(over_text, (screen_width // 2 - over_text.get_width() // 2, 150))
            screen.blit(score_text, (screen_width // 2 - score_text.get_width() // 2, 220))
            screen.blit(restart_text, (screen_width // 2 - restart_text.get_width() // 2, 280))

        pygame.display.flip()
        clock.tick(30)  # Limit to 30 FPS

    cap.release()
    pygame.quit()

if __name__ == "__main__":
    main()
