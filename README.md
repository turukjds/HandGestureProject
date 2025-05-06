# Hand Gesture 3D Manipulation Project (Soccer Demo)

A Python project utilizing computer vision (OpenCV, MediaPipe) and a 3D game engine (Ursina) to allow users to interact with a virtual soccer ball using hand gestures detected via webcam. Users can "pinch" to grab the ball and "throw" it by releasing the pinch while moving their hand.

## Overview

This project captures video input from a webcam, processes frames to detect hand landmarks using MediaPipe, analyzes the proximity of the thumb and index finger tips to detect a "pinch" gesture, and translates hand movements into forces applied to a virtual soccer ball within a 3D environment built with Ursina. The goal is to grab the ball by pinching near it on screen, move it by moving the pinched hand, and score goals by throwing the ball into nets.

## How it Works

The core logic combines computer vision for gesture input and 3D physics simulation for interaction:

1.  **Initialization:**
    *   **Ursina Setup:** Initializes the Ursina engine, sets up the window, camera, and creates the 3D environment including a soccer field (`plane`), goals (visual posts/crossbar and invisible trigger colliders), and the soccer ball (`sphere`) entity with initial properties. UI elements like the score display (`Text`) and a reset button (`Button`) are also created.
    *   **Webcam & MediaPipe:** Initializes OpenCV `VideoCapture` to access the default webcam. Sets up the MediaPipe Hands solution (`mp.solutions.hands`) configured to detect a single hand with specific confidence thresholds.
    *   **State Variables:** Initializes variables to track the interaction state, such as whether the user is currently pinching (`is_pinching`), which object is grabbed (`grabbed_object`), the screen and world coordinates at the start of a pinch, etc.

2.  **Main Update Loop (`update` function in Ursina):** This function runs every frame.
    *   **Video Capture & Hand Detection:** Reads a frame from the webcam, flips it horizontally (for a mirror view), converts color space (BGR to RGB) for MediaPipe, and processes it using `hands.process()` to find hand landmarks. The image is converted back to BGR for display.
    *   **Landmark Processing (if hand detected):**
        *   Extracts the 3D landmarks for the detected hand.
        *   Specifically gets the coordinates of the `THUMB_TIP` and `INDEX_FINGER_TIP`.
        *   Calculates the Euclidean distance between these two landmarks.
        *   Calculates the center point between the tips (pinch center) in normalized image coordinates (0.0 to 1.0).
        *   Converts this normalized pinch center to Ursina's screen coordinate system (where the center is (0,0) and edges are +/- 0.5 * aspect_ratio horizontally and +/- 0.5 vertically).
    *   **Pinch Gesture Logic:**
        *   **Pinch Detection:** If the calculated distance between thumb and index tips is less than `PINCH_THRESHOLD`, a pinch is detected for this frame.
        *   **Pinch Start (Grab):** If a pinch is detected *and* the user wasn't already pinching (`not is_pinching`):
            *   It checks the proximity of the pinch gesture to the ball *on the screen*. It calculates the squared distance between the ball's `screen_position` and the `current_pinch_pos_screen`.
            *   If this screen distance is below a small threshold (meaning the pinch happened *near* the ball visually), the grab is initiated:
                *   `is_pinching` is set to `True`.
                *   `grabbed_object` is set to the `ball` entity.
                *   The initial screen position of the pinch and the initial world position of the ball are stored.
                *   The hand position smoother is initialized.
        *   **Pinch Held (Move/Pull):** If a pinch is detected *and* the user is already pinching the ball (`is_pinching and grabbed_object == ball`):
            *   The current screen position of the pinch is smoothed using an exponential moving average (`SMOOTHING_FACTOR`).
            *   The difference (delta) between the current smoothed screen position and the initial pinch screen position is calculated.
            *   This screen delta is scaled (`MOVE_SCALE`) and added to the ball's initial *world* position to determine a target world position for the ball.
            *   **Crucially:** Instead of teleporting the ball, a "pull" force is calculated. This force is a vector pointing from the ball's current position towards the calculated target position.
            *   This pull force (scaled by `PULL_STRENGTH` and `time.dt`) is added to the ball's `velocity`. This allows the ball to follow the hand while still being affected by physics (like gravity).
        *   **Pinch Release (Throw):** If the distance between tips is *greater* than `PINCH_THRESHOLD`:
            *   If the user *was* pinching, a release counter (`pinch_release_buffer`) is incremented.
            *   Once this counter exceeds `PINCH_RELEASE_DELAY` (a few frames to prevent accidental releases), the pinch state is fully reset (`is_pinching = False`, `grabbed_object = None`). The ball continues moving with the velocity it had due to the pull force, simulating a throw.
    *   **Ball Physics Simulation (Always Active):**
        *   Gravity (`GRAVITY`) is applied to the ball's vertical velocity (`ball.velocity.y`).
        *   The ball's `position` is updated based on its current `velocity` and `time.dt`.
        *   A visual rolling effect is applied to the ball's `rotation` if it's near the ground and moving horizontally.
        *   **Ground Collision:** Checks if the bottom of the ball hits the field's Y-level. If so:
            *   Corrects the ball's position to be exactly on the ground.
            *   Reverses and dampens the vertical velocity based on `RESTITUTION` (bounce).
            *   Reduces the horizontal velocity based on `FRICTION`.
            *   Very small velocities are zeroed out to prevent jittering.
    *   **Boundary & Goal Checks:**
        *   Checks if the ball has fallen too far below the field or gone too far outside the horizontal boundaries. If so, the ball is reset to the center using `reset_ball_function`.
        *   Checks if the ball `intersects` with the invisible `goal_1_trigger` or `goal_2_trigger` entities. If an intersection occurs:
            *   The score is incremented, the UI text is updated.
            *   The ball is reset using `reset_ball_function`.
    *   **Display & Quit:**
        *   The OpenCV window (`Hand Tracking Debug`) displays the webcam feed with the detected hand landmarks drawn.
        *   Listens for the 'q' key press in the OpenCV window to trigger `app.quit()`, stopping the Ursina application.

3.  **Cleanup:** When the application quits (either via 'q' key or closing the Ursina window), it releases the webcam resource (`cap.release()`) and destroys all OpenCV windows.

## Libraries Used

*   **OpenCV (`opencv-python`):** For video capture from webcam and basic image processing/display.
*   **MediaPipe (`mediapipe`):** For high-fidelity hand tracking and landmark detection.
*   **Ursina (`ursina`):** A Python game engine used for creating the 3D environment, handling entity rendering, physics simulation (custom in this case), and user interaction loop.
*   **math:** For mathematical calculations (distance, square root).
*   **time:** Used for `time.dt` (delta time) in physics calculations within the `update` loop.
*   **random:** Used for randomizing ball reset position slightly.

## Setup and Usage

1.  **Prerequisites:** Python 3.x installed.
2.  **Clone the repository:**
    ```bash
    git clone https://github.com/turukjds/HandGestureProject.git
    cd HandGestureProject
    ```
3.  **Install dependencies:**
    ```bash
    pip install opencv-python mediapipe ursina
    ```
4.  **Run the script:**
    ```bash
    python handgesture_3d_manipulation.py
    ```
5.  **Interaction:**
    *   Ensure your hand is visible to the webcam.
    *   Make a "pinch" gesture (bring thumb and index finger tips close) near the ball on the screen to grab it.
    *   Move your pinched hand to move the ball.
    *   Release the pinch while moving your hand to "throw" the ball.
    *   Try to score in the goals!
    *   Press 'q' in the OpenCV debug window or close the Ursina window to quit.
    *   Click the "Reset Ball" button in the Ursina window if the ball gets stuck or lost.

## Recognized Gestures & Actions

*   **Pinch:** Bringing the thumb tip and index finger tip close together (distance < `PINCH_THRESHOLD`).
    *   **Action:** If performed near the ball on the screen, initiates a "grab" of the ball.
*   **Move Pinched Hand:** Moving the hand while maintaining the pinch gesture after grabbing the ball.
    *   **Action:** Applies a force to the grabbed ball, pulling it towards the corresponding world position of the hand's screen movement.
*   **Release Pinch:** Separating the thumb and index finger tips after pinching.
    *   **Action:** Releases the ball, allowing it to continue moving with the velocity it gained from the pulling force (simulating a throw).
