import cv2
import mediapipe as mp
import math
from ursina import *
import random
import time # For velocity calculation

# --- Constants ---
PINCH_THRESHOLD = 0.06      # Distance between thumb and index finger to trigger pinch
# GRAB_DISTANCE_THRESHOLD = 1.5 # Max 3D distance from hand to ball to initiate grab (Using screen check now)
MOVE_SCALE = 25             # Sensitivity of ball movement when grabbed
# THROW_STRENGTH = 1.5        # Multiplier for throw velocity (Not needed for this method)
SMOOTHING_FACTOR = 0.4      # Smoothing for hand movement (0.0-1.0, lower is smoother)
# HISTORY_LENGTH = 5          # Number of past positions to track for throw velocity (Not needed)
PINCH_RELEASE_DELAY = 3     # Frames pinch must be released before triggering release action
PULL_STRENGTH = 10          # How strongly the hand pulls the ball (adjust this)

GRAVITY = -15.0             # Acceleration due to gravity
RESTITUTION = 0.6           # Bounciness (0.0 - 1.0)
FRICTION = 0.95             # Slowdown factor on ground (lower is more friction)

FIELD_WIDTH = 20
FIELD_LENGTH = 30
GOAL_WIDTH = 4
GOAL_HEIGHT = 2
GOAL_DEPTH = 1.0

# --- Helper Functions ---
def distance_3d(v1, v2):
	"""Calculates the Euclidean distance between two Vec3 objects."""
	return math.sqrt((v1.x - v2.x)**2 + (v1.y - v2.y)**2 + (v1.z - v2.z)**2)

# --- Ursina Setup ---
app = Ursina()
window.title = 'Hand Gesture Soccer'
window.borderless = False
window.exit_button.visible = False
window.vsync = True
# window.fps_counter.enabled = True # Uncomment for debugging

# --- Game Entities ---
field = Entity(model='plane', scale=(FIELD_WIDTH, 1, FIELD_LENGTH), color=color.green.tint(-0.2), collider='box')
field_y = 0 # Ground level

ball = Entity(model='sphere', color=color.white, scale=0.5, collider='sphere', position=(0, 2, 0))
ball.velocity = Vec3(0, 0, 0)
ball.rotation_speed = Vec3(0, 0, 0) # For rolling effect
# ball_physics_enabled = True # Physics is always enabled now

# Goals
goal_y_pos = field_y + GOAL_HEIGHT / 2
goal_z_pos = field.scale_z / 2 # Position goals right at the edge

# Goal 1 (Negative Z) - Visuals
goal_1_post_left = Entity(model='cube', scale=(0.1, GOAL_HEIGHT, 0.1), position=(-GOAL_WIDTH/2, goal_y_pos, -goal_z_pos), color=color.white)
goal_1_post_right = Entity(model='cube', scale=(0.1, GOAL_HEIGHT, 0.1), position=(GOAL_WIDTH/2, goal_y_pos, -goal_z_pos), color=color.white)
goal_1_crossbar = Entity(model='cube', scale=(GOAL_WIDTH, 0.1, 0.1), position=(0, field_y + GOAL_HEIGHT, -goal_z_pos), color=color.white)
# Goal 1 - Trigger (Slightly behind visuals, larger)
goal_1_trigger = Entity(model='cube', scale=(GOAL_WIDTH * 1.1, GOAL_HEIGHT * 1.1, GOAL_DEPTH), position=(0, goal_y_pos, -goal_z_pos - GOAL_DEPTH/2), collider='box', visible=False)

# Goal 2 (Positive Z) - Visuals
goal_2_post_left = Entity(model='cube', scale=(0.1, GOAL_HEIGHT, 0.1), position=(-GOAL_WIDTH/2, goal_y_pos, goal_z_pos), color=color.white)
goal_2_post_right = Entity(model='cube', scale=(0.1, GOAL_HEIGHT, 0.1), position=(GOAL_WIDTH/2, goal_y_pos, goal_z_pos), color=color.white)
goal_2_crossbar = Entity(model='cube', scale=(GOAL_WIDTH, 0.1, 0.1), position=(0, field_y + GOAL_HEIGHT, goal_z_pos), color=color.white)
# Goal 2 - Trigger (Slightly behind visuals, larger)
goal_2_trigger = Entity(model='cube', scale=(GOAL_WIDTH * 1.1, GOAL_HEIGHT * 1.1, GOAL_DEPTH), position=(0, goal_y_pos, goal_z_pos + GOAL_DEPTH/2), collider='box', visible=False)

# --- UI ---
score = 0
score_text = Text(text=f"Score: {score}", origin=(-0.5, 0.5), scale=2, position=window.top_left + Vec2(0.05, -0.05))

def reset_ball_function():
	"""Resets the ball to the center."""
	global is_pinching, grabbed_object # Removed ball_physics_enabled
	global initial_pinch_pos_screen, initial_object_pos, pinch_history, last_smoothed_pinch_pos, pinch_release_buffer

	ball.position = (random.uniform(-2, 2), 2, random.uniform(-2, 2)) # Reset near center
	ball.velocity = Vec3(0, 0, 0)
	ball.rotation_speed = Vec3(0, 0, 0)
	# ball_physics_enabled = True # Physics always on

	# Reset grab state if ball was grabbed during reset
	if is_pinching and grabbed_object == ball:
		is_pinching = False
		grabbed_object = None
		initial_pinch_pos_screen = None
		initial_object_pos = None
		# pinch_history = [] # Not needed for this method
		last_smoothed_pinch_pos = None
		pinch_release_buffer = 0

# Position the button at the top-middle
reset_button = Button(
	text='Reset Ball',
	scale=(0.2, 0.05),
	origin=(0, 0.5), # Set origin to top-center for centering
	position=window.top + Vec2(0, -0.05), # Position at top-center, slightly offset down
	color=color.azure,
	on_click=reset_ball_function
)

# --- Camera ---
camera.position = (0, 15, -25) # Adjusted position
camera.rotation_x = 30
camera.fov = 70 # Slightly wider FOV

# --- Webcam & Hand Tracking Setup ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
	print("Error: Could not open webcam.")
	app.quit()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
	static_image_mode=False,
	max_num_hands=1,            # Track only one hand
	min_detection_confidence=0.7,
	min_tracking_confidence=0.7
)

# --- Interaction State Variables ---
is_pinching = False
grabbed_object = None
initial_pinch_pos_screen = None # Stores the Vec3 screen coordinate where pinch started
initial_object_pos = None     # Stores the Vec3 world coordinate of the object when grabbed
# pinch_history = []            # List of {'pos': Vec3, 'time': float} for velocity calculation (Not needed)
last_smoothed_pinch_pos = None # For smoothing movement
pinch_release_buffer = 0      # Counter for release delay

# --- Main Update Loop ---
def update():
	# Declare globals that are modified
	global is_pinching, grabbed_object, score # Removed ball_physics_enabled
	global initial_pinch_pos_screen, initial_object_pos, last_smoothed_pinch_pos, pinch_release_buffer # Removed pinch_history
	global reset_ball # Used for boundary checks

	reset_ball = False # Reset flag for boundary checks each frame

	# --- Hand Tracking ---
	success, image = cap.read()
	if not success:
		return # Skip frame if webcam read fails

	# Flip the image horizontally for a later selfie-view display, and convert
	image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
	image.flags.writeable = False # To improve performance
	results = hands.process(image)
	image.flags.writeable = True # Image is now writeable again
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Convert back to BGR for OpenCV display

	pinch_detected_this_frame = False
	current_pinch_pos_screen = None # Ursina screen coordinates (-0.5 to 0.5)

	if results.multi_hand_landmarks:
		# --- Process Detected Hand ---
		hand_landmarks = results.multi_hand_landmarks[0] # Assuming max_num_hands=1
		mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

		# Get thumb and index finger tip coordinates (normalized 0.0-1.0)
		thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
		index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

		# Calculate distance between tips
		distance = math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2 + (thumb_tip.z - index_tip.z)**2)

		# Calculate pinch center in normalized coordinates
		pinch_center_x_norm = (thumb_tip.x + index_tip.x) / 2
		pinch_center_y_norm = (thumb_tip.y + index_tip.y) / 2

		# Convert pinch center to Ursina screen coordinates (center is 0,0)
		current_pinch_pos_screen = Vec3(pinch_center_x_norm - 0.5, (1.0 - pinch_center_y_norm) - 0.5, 0)

		# --- Pinch State Logic ---
		if distance < PINCH_THRESHOLD:
			pinch_detected_this_frame = True
			pinch_release_buffer = 0 # Reset release buffer

			# --- Pinch Start ---
			if not is_pinching:
				# Check proximity to the ball using screen position
				screen_dist_sq = (ball.screen_position.x - current_pinch_pos_screen.x)**2 + \
								 (ball.screen_position.y - current_pinch_pos_screen.y)**2

				# Check if pinch is close on screen
				if screen_dist_sq < 0.015: # Adjust threshold as needed
					print("Grabbed Ball!")
					is_pinching = True
					grabbed_object = ball
					# ball_physics_enabled = False # Physics stays ON
					# ball.velocity = Vec3(0, 0, 0) # Don't stop ball immediately, let pull handle it
					ball.rotation_speed = Vec3(0, 0, 0) # Stop visual roll
					initial_pinch_pos_screen = current_pinch_pos_screen # Store screen pos
					initial_object_pos = Vec3(ball.position)      # Store world pos
					last_smoothed_pinch_pos = current_pinch_pos_screen # Initialize smoother
					# pinch_history = [] # Not needed

		else: # Pinch not detected
			if is_pinching:
				pinch_release_buffer += 1 # Increment release buffer

		# --- Pinch Held ---
		if is_pinching and grabbed_object == ball:
			# Apply smoothing to the current pinch screen position
			if last_smoothed_pinch_pos is None:
				last_smoothed_pinch_pos = current_pinch_pos_screen # Initialize on first frame
			smoothed_pinch_pos = lerp(last_smoothed_pinch_pos, current_pinch_pos_screen, SMOOTHING_FACTOR)
			last_smoothed_pinch_pos = smoothed_pinch_pos # Update for next frame

			# Calculate the change in screen position from the start
			delta_pos_screen = smoothed_pinch_pos - initial_pinch_pos_screen

			# Calculate target world position (Constrained to X/Z plane near ground)
			target_world_pos = Vec3(
				initial_object_pos.x + delta_pos_screen.x * MOVE_SCALE,         # Screen X -> World X
				field_y + ball.scale_y / 2,                                   # Target Y fixed near ground
				initial_object_pos.z - delta_pos_screen.y * MOVE_SCALE          # Screen Y -> World Z (Forward/Backward)
			)

			# Calculate difference vector from ball to target
			diff_vector = target_world_pos - ball.position

			# Apply velocity to pull the ball towards the target position (Primarily Horizontal)
			# This velocity will fight against friction and inertia, while gravity handles Y
			pull_velocity = diff_vector * PULL_STRENGTH
			ball.velocity.x = pull_velocity.x # Apply horizontal pull
			ball.velocity.z = pull_velocity.z # Apply depth pull
			# Let gravity and ground collision manage ball.velocity.y

			# Optional: Clamp target position to field bounds
			# target_world_pos.x = clamp(target_world_pos.x, -FIELD_WIDTH/2, FIELD_WIDTH/2)
			# target_world_pos.z = clamp(target_world_pos.z, -FIELD_LENGTH/2, FIELD_LENGTH/2)


	# --- Pinch Release ---
	# Trigger release only if pinch was active, is no longer detected, and buffer time passed
	if is_pinching and not pinch_detected_this_frame and pinch_release_buffer >= PINCH_RELEASE_DELAY:
		print("Pinch Released")
		is_pinching = False
		# ball_physics_enabled = True # Physics was always on

		# --- NO Throw Velocity Calculation Needed ---
		# The ball continues with the velocity it had from the pull effect

		# Reset grab state variables completely
		grabbed_object = None
		initial_pinch_pos_screen = None
		initial_object_pos = None
		# pinch_history = [] # Not needed
		last_smoothed_pinch_pos = None
		pinch_release_buffer = 0

	# --- Ball Physics Simulation (Always Runs) ---
	# Apply gravity
	ball.velocity.y += GRAVITY * time.dt # Gravity still affects the ball

	# Apply rotation (rolling) - This is visual, doesn't affect physics directly here
	# Only apply if ball is on/near ground and moving horizontally
	if ball.y < field_y + ball.scale_y and ball.velocity.xz.length() > 0.1:
		roll_direction = Vec3(ball.velocity.z, 0, -ball.velocity.x).normalized()
		ball.rotation_speed = roll_direction * ball.velocity.length() * 2 # Adjust multiplier
		ball.rotation += ball.rotation_speed * time.dt
	else:
		ball.rotation_speed = Vec3(0,0,0) # Stop visual roll if in air or stopped


	# Update position based on velocity
	ball.position += ball.velocity * time.dt

	# Collision with ground
	ball_bottom_y = ball.y - ball.scale_y / 2
	if ball_bottom_y <= field_y:
		ball.y = field_y + ball.scale_y / 2 # Correct position
		# Apply bounce (restitution) and friction only if moving downwards
		if ball.velocity.y < 0:
			ball.velocity.y *= -RESTITUTION # Bounce
			# Apply friction to horizontal velocity
			ball.velocity.x *= FRICTION
			ball.velocity.z *= FRICTION

			# Stop bounce/roll if velocity/speed is very low
			if abs(ball.velocity.y) < 0.1:
				ball.velocity.y = 0
			if ball.velocity.xz.length() < 0.1:
				ball.velocity.x = 0
				ball.velocity.z = 0


	# --- Boundary Check ---
	if ball.y < field_y - 5: # Fell through floor
		print("Ball fell too low, resetting.")
		reset_ball = True
	elif abs(ball.x) > FIELD_WIDTH / 2 + 2 or abs(ball.z) > FIELD_LENGTH / 2 + 2: # Went far out
		print("Ball went out of bounds, resetting.")
		reset_ball = True

	if reset_ball:
		reset_ball_function() # Use the reset function

	# --- Goal Detection ---
	scored = False
	# Goal check doesn't need 'not is_pinching' because physics is always on.
	# If you manage to drag the ball into the goal, it should count.
	if not reset_ball: # Only check if ball wasn't just reset
		if ball.intersects(goal_1_trigger).hit:
			print("GOAL! Goal 1!")
			score += 1
			scored = True
		elif ball.intersects(goal_2_trigger).hit:
			print("GOAL! Goal 2!")
			score += 1
			scored = True

	if scored:
		score_text.text = f"Score: {score}"
		reset_ball_function() # Use the reset function

	# --- Display Debug Window ---
	cv2.imshow('Hand Tracking Debug', image)
	if cv2.waitKey(5) & 0xFF == ord('q'):
		app.quit() # Quit Ursina app
		cap.release()
		cv2.destroyAllWindows()

# --- Start Ursina ---
app.run()

# --- Cleanup ---
print("Cleaning up...")
if cap.isOpened():
	cap.release()
cv2.destroyAllWindows()
print("Application finished.")
