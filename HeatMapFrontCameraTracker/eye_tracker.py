"""Self-contained 3D eye tracker used by HeatMapFrontCameraTracker."""

import cv2
import random
import math
import numpy as np
import os
import time

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
GAZE_VECTOR_PATH = os.path.join(MODULE_DIR, "gaze_vector.txt")

try:
    import gl_sphere
    GL_SPHERE_AVAILABLE = True
except ImportError:
    GL_SPHERE_AVAILABLE = False
    print("gl_sphere module not found. OpenGL rendering will be disabled.")

EYE_IDS = ("left", "right")
current_eye_id = "left"

ray_lines = []
model_centers = []
max_rays = 100
prev_model_center_avg = (320, 240)
max_observed_distance = 0
stored_intersections = []

last_sphere_center = None
last_gaze_dir = None
calibrated_sphere_center = None
sphere_center_locked_2d = False
locked_model_center_avg = prev_model_center_avg

# Combined gaze (fused from left + right) used by heatmap / front camera
combined_gaze_dir = None

calibrated = False
R_gaze_to_cam = np.eye(3, dtype=np.float32)

EXT_WIDTH = 640
EXT_HEIGHT = 480
EXT_CX = EXT_WIDTH // 2
EXT_CY = EXT_HEIGHT // 2
EXT_FX = 600.0
EXT_FY = 600.0
circle_x = EXT_CX
circle_y = EXT_CY

# Exponential smoothing for fused gaze (heatmap + front-camera dot).
GAZE_DIRECTION_SMOOTH_ALPHA = 0.22
GAZE_CIRCLE_SMOOTH_ALPHA = 0.35
smoothed_combined_gaze_dir = None
_smoothed_circle_u = None
_smoothed_circle_v = None


def create_eye_tracking_state():
    return {
        "ray_lines": [],
        "model_centers": [],
        "prev_model_center_avg": (320, 240),
        "max_observed_distance": 0,
        "stored_intersections": [],
        "last_sphere_center": None,
        "last_gaze_dir": None,
        "calibrated_sphere_center": None,
        "sphere_center_locked_2d": False,
        "locked_model_center_avg": (320, 240),
        "gaze_output": [0.0] * 6,
    }


eye_tracking_states = {
    eye_id: create_eye_tracking_state()
    for eye_id in EYE_IDS
}


def eye_window_name(base_name):
    return f"{current_eye_id} {base_name}"


def load_eye_tracking_state(eye_id):
    global current_eye_id
    global ray_lines, model_centers, prev_model_center_avg, max_observed_distance
    global stored_intersections, last_sphere_center, last_gaze_dir
    global calibrated_sphere_center, sphere_center_locked_2d, locked_model_center_avg

    current_eye_id = eye_id
    state = eye_tracking_states[eye_id]
    ray_lines = state["ray_lines"]
    model_centers = state["model_centers"]
    prev_model_center_avg = state["prev_model_center_avg"]
    max_observed_distance = state["max_observed_distance"]
    stored_intersections = state["stored_intersections"]
    last_sphere_center = state["last_sphere_center"]
    last_gaze_dir = state["last_gaze_dir"]
    calibrated_sphere_center = state["calibrated_sphere_center"]
    sphere_center_locked_2d = state["sphere_center_locked_2d"]
    locked_model_center_avg = state["locked_model_center_avg"]


def save_eye_tracking_state(eye_id):
    state = eye_tracking_states[eye_id]
    state["ray_lines"] = ray_lines
    state["model_centers"] = model_centers
    state["prev_model_center_avg"] = prev_model_center_avg
    state["max_observed_distance"] = max_observed_distance
    state["stored_intersections"] = stored_intersections
    state["last_sphere_center"] = last_sphere_center
    state["last_gaze_dir"] = last_gaze_dir
    state["calibrated_sphere_center"] = calibrated_sphere_center
    state["sphere_center_locked_2d"] = sphere_center_locked_2d
    state["locked_model_center_avg"] = locked_model_center_avg


def reset_eye_tracking_state(eye_id):
    eye_tracking_states[eye_id] = create_eye_tracking_state()
    write_gaze_vector_file()
    if current_eye_id == eye_id:
        load_eye_tracking_state(eye_id)


def get_eye_gaze_dir(eye_id):
    direction = eye_tracking_states[eye_id]["last_gaze_dir"]
    if direction is None:
        return None
    direction = np.asarray(direction, dtype=np.float32)
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return None
    return direction / norm


def combine_gaze_directions(*directions):
    valid = []
    for direction in directions:
        if direction is None:
            continue
        direction = np.asarray(direction, dtype=np.float32)
        norm = np.linalg.norm(direction)
        if norm >= 1e-6:
            valid.append(direction / norm)

    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]

    combined = np.sum(valid, axis=0)
    norm = np.linalg.norm(combined)
    if norm < 1e-6:
        return valid[0]
    return combined / norm


def reset_gaze_smoothing():
    global smoothed_combined_gaze_dir, _smoothed_circle_u, _smoothed_circle_v
    smoothed_combined_gaze_dir = None
    _smoothed_circle_u = None
    _smoothed_circle_v = None


def smooth_combined_gaze(direction, alpha=GAZE_DIRECTION_SMOOTH_ALPHA):
    """EMA on the fused unit gaze direction; holds last value when tracking drops."""
    global smoothed_combined_gaze_dir

    if direction is None:
        return smoothed_combined_gaze_dir

    direction = np.asarray(direction, dtype=np.float32)
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return smoothed_combined_gaze_dir
    direction = direction / norm

    if smoothed_combined_gaze_dir is None:
        smoothed_combined_gaze_dir = direction.copy()
        return smoothed_combined_gaze_dir

    blended = (1.0 - alpha) * smoothed_combined_gaze_dir + alpha * direction
    blend_norm = np.linalg.norm(blended)
    if blend_norm < 1e-6:
        return smoothed_combined_gaze_dir

    smoothed_combined_gaze_dir = (blended / blend_norm).astype(np.float32)
    return smoothed_combined_gaze_dir


def refresh_combined_gaze(active_eyes=EYE_IDS):
    global combined_gaze_dir, last_gaze_dir

    directions = [get_eye_gaze_dir(eye_id) for eye_id in active_eyes]
    combined_gaze_dir = combine_gaze_directions(*directions)
    last_gaze_dir = combined_gaze_dir
    return smooth_combined_gaze(combined_gaze_dir)


def get_combined_gaze_dir():
    if smoothed_combined_gaze_dir is not None:
        return smoothed_combined_gaze_dir
    return combined_gaze_dir if combined_gaze_dir is not None else last_gaze_dir


def get_raw_combined_gaze_dir():
    return combined_gaze_dir if combined_gaze_dir is not None else last_gaze_dir


def write_gaze_vector_file():
    file_path = GAZE_VECTOR_PATH

    def is_file_available(path):
        try:
            with open(path, "a"):
                return True
        except IOError:
            return False

    if not is_file_available(file_path):
        print("File is currently in use. Skipping write.")
        return

    try:
        values = eye_tracking_states["left"]["gaze_output"] + eye_tracking_states["right"]["gaze_output"]
        csv_line = ",".join(f"{value:.6f}" for value in values)
        with open(file_path, "w") as f:
            f.write(csv_line + "\n")
    except Exception as e:
        print("Write error:", e)


def update_eye_gaze_output(eye_id, sphere_center, gaze_direction):
    all_values = np.concatenate((sphere_center, gaze_direction))
    eye_tracking_states[eye_id]["gaze_output"] = [float(value) for value in all_values]
    write_gaze_vector_file()


def configure_external_viewport(width, height):
    global EXT_WIDTH, EXT_HEIGHT, EXT_CX, EXT_CY, EXT_FX, EXT_FY, circle_x, circle_y

    EXT_WIDTH = max(int(width), 1)
    EXT_HEIGHT = max(int(height), 1)
    EXT_CX = EXT_WIDTH // 2
    EXT_CY = EXT_HEIGHT // 2
    EXT_FX = 600.0 * (EXT_WIDTH / 640.0)
    EXT_FY = 600.0 * (EXT_HEIGHT / 480.0)
    circle_x, circle_y = EXT_CX, EXT_CY


def open_external_camera(external_index, preferred_width=640, preferred_height=480):
    if external_index is None:
        return None

    external_cap = cv2.VideoCapture(external_index, cv2.CAP_MSMF)
    if not external_cap.isOpened():
        print(f"Warning: Could not open external camera at index {external_index}.")
        return None

    external_cap.set(cv2.CAP_PROP_FRAME_WIDTH, preferred_width)
    external_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, preferred_height)

    ret, frame = external_cap.read()
    if ret and frame is not None:
        frame_height, frame_width = frame.shape[:2]
        configure_external_viewport(frame_width, frame_height)
    else:
        configure_external_viewport(preferred_width, preferred_height)

    print(f"External camera opened at index {external_index} ({EXT_WIDTH}x{EXT_HEIGHT}).")
    return external_cap


# Function to detect available cameras (gentle probe — avoids USB reset storms)
def detect_cameras(max_cams=6):
    available_cameras = []
    for i in range(max_cams):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
        time.sleep(0.2)
    return available_cameras

# Crop the image to maintain a specific aspect ratio (width:height) before resizing.
def crop_to_aspect_ratio(image, width=640, height=480):
    current_height, current_width = image.shape[:2]
    desired_ratio = width / height
    current_ratio = current_width / current_height

    if current_ratio > desired_ratio:
        # Current image is too wide
        new_width = int(desired_ratio * current_height)
        offset = (current_width - new_width) // 2
        cropped_img = image[:, offset:offset + new_width]
    else:
        # Current image is too tall
        new_height = int(current_width / desired_ratio)
        offset = (current_height - new_height) // 2
        cropped_img = image[offset:offset + new_height, :]

    return cv2.resize(cropped_img, (width, height))

# Apply thresholding to an image
def apply_binary_threshold(image, darkestPixelValue, addedThreshold):
    threshold = darkestPixelValue + addedThreshold
    _, thresholded_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    return thresholded_image

# Finds a square area of dark pixels in the image
def get_darkest_area(image):
    ignoreBounds = 20
    imageSkipSize = 10
    searchArea = 20
    internalSkipSize = 5

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_sum = float('inf')
    darkest_point = None

    for y in range(ignoreBounds, gray.shape[0] - ignoreBounds, imageSkipSize):
        for x in range(ignoreBounds, gray.shape[1] - ignoreBounds, imageSkipSize):
            current_sum = 0
            num_pixels = 0
            for dy in range(0, searchArea, internalSkipSize):
                if y + dy >= gray.shape[0]:
                    break
                for dx in range(0, searchArea, internalSkipSize):
                    if x + dx >= gray.shape[1]:
                        break
                    current_sum += gray[y + dy][x + dx]
                    num_pixels += 1

            if current_sum < min_sum and num_pixels > 0:
                min_sum = current_sum
                darkest_point = (x + searchArea // 2, y + searchArea // 2)

    return darkest_point

# Mask all pixels outside a square defined by center and size
def mask_outside_square(image, center, size):
    x, y = center
    half_size = size // 2

    mask = np.zeros_like(image)
    top_left_x = max(0, x - half_size)
    top_left_y = max(0, y - half_size)
    bottom_right_x = min(image.shape[1], x + half_size)
    bottom_right_y = min(image.shape[0], y + half_size)
    mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 255
    return cv2.bitwise_and(image, mask)

def optimize_contours_by_angle(contours, image):
    if len(contours) < 1:
        return contours

    # Holds the candidate points
    all_contours = np.concatenate(contours[0], axis=0)

    # Set spacing based on size of contours
    spacing = int(len(all_contours)/25)  # Spacing between sampled points

    # Temporary array for result
    filtered_points = []
    
    # Calculate centroid of the original contours
    centroid = np.mean(all_contours, axis=0)
    
    # Create an image of the same size as the original image
    point_image = image.copy()
    
    skip = 0
    
    # Loop through each point in the all_contours array
    for i in range(0, len(all_contours), 1):
    
        # Get three points: current point, previous point, and next point
        current_point = all_contours[i]
        prev_point = all_contours[i - spacing] if i - spacing >= 0 else all_contours[-spacing]
        next_point = all_contours[i + spacing] if i + spacing < len(all_contours) else all_contours[spacing]
        
        # Calculate vectors between points
        vec1 = prev_point - current_point
        vec2 = next_point - current_point
        
        with np.errstate(invalid='ignore'):
            # Calculate angles between vectors
            angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

        
        # Calculate vector from current point to centroid
        vec_to_centroid = centroid - current_point
        
        # Check if angle is oriented towards centroid
        # Calculate the cosine of the desired angle threshold (e.g., 80 degrees)
        cos_threshold = np.cos(np.radians(60))  # Convert angle to radians
        
        if np.dot(vec_to_centroid, (vec1+vec2)/2) >= cos_threshold:
            filtered_points.append(current_point)
    
    return np.array(filtered_points, dtype=np.int32).reshape((-1, 1, 2))

# Returns the largest contour that is not extremely long or tall
def filter_contours_by_area_and_return_largest(contours, pixel_thresh, ratio_thresh):
    max_area = 0
    largest_contour = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= pixel_thresh:
            x, y, w, h = cv2.boundingRect(contour)
            length_to_width_ratio = max(w / h, h / w)
            if length_to_width_ratio <= ratio_thresh:
                if area > max_area:
                    max_area = area
                    largest_contour = contour

    return [largest_contour] if largest_contour is not None else []
#Fits an ellipse to the optimized contours and draws it on the image.
def fit_and_draw_ellipses(image, optimized_contours, color):
    if len(optimized_contours) >= 5:
        # Ensure the data is in the correct shape (n, 1, 2) for cv2.fitEllipse
        contour = np.array(optimized_contours, dtype=np.int32).reshape((-1, 1, 2))

        # Fit ellipse
        ellipse = cv2.fitEllipse(contour)

        # Draw the ellipse
        cv2.ellipse(image, ellipse, color, 2)  # Draw with green color and thickness of 2

        return image
    else:
        print("Not enough points to fit an ellipse.")
        return image

#checks how many pixels in the contour fall under a slightly thickened ellipse
#also returns that number of pixels divided by the total pixels on the contour border
#assists with checking ellipse goodness    
def check_contour_pixels(contour, image_shape, debug_mode_on):
    # Check if the contour can be used to fit an ellipse (requires at least 5 points)
    if len(contour) < 5:
        return [0, 0]  # Not enough points to fit an ellipse
    
    # Create an empty mask for the contour
    contour_mask = np.zeros(image_shape, dtype=np.uint8)
    # Draw the contour on the mask, filling it
    cv2.drawContours(contour_mask, [contour], -1, (255), 1)
   
    # Fit an ellipse to the contour and create a mask for the ellipse
    ellipse_mask_thick = np.zeros(image_shape, dtype=np.uint8)
    ellipse_mask_thin = np.zeros(image_shape, dtype=np.uint8)
    ellipse = cv2.fitEllipse(contour)
    
    # Draw the ellipse with a specific thickness
    cv2.ellipse(ellipse_mask_thick, ellipse, (255), 10) #capture more for absolute
    cv2.ellipse(ellipse_mask_thin, ellipse, (255), 4) #capture fewer for ratio

    # Calculate the overlap of the contour mask and the thickened ellipse mask
    overlap_thick = cv2.bitwise_and(contour_mask, ellipse_mask_thick)
    overlap_thin = cv2.bitwise_and(contour_mask, ellipse_mask_thin)
    
    # Count the number of non-zero (white) pixels in the overlap
    absolute_pixel_total_thick = np.sum(overlap_thick > 0)#compute with thicker border
    absolute_pixel_total_thin = np.sum(overlap_thin > 0)#compute with thicker border
    
    # Compute the ratio of pixels under the ellipse to the total pixels on the contour border
    total_border_pixels = np.sum(contour_mask > 0)
    
    ratio_under_ellipse = absolute_pixel_total_thin / total_border_pixels if total_border_pixels > 0 else 0
    
    return [absolute_pixel_total_thick, ratio_under_ellipse, overlap_thin]

#outside of this method, select the ellipse with the highest percentage of pixels under the ellipse 
#TODO for efficiency, work with downscaled or cropped images
def check_ellipse_goodness(binary_image, contour, debug_mode_on):
    ellipse_goodness = [0,0,0] #covered pixels, edge straightness stdev, skewedness   
    # Check if the contour can be used to fit an ellipse (requires at least 5 points)
    if len(contour) < 5:
        print("length of contour was 0")
        return 0  # Not enough points to fit an ellipse
    
    # Fit an ellipse to the contour
    ellipse = cv2.fitEllipse(contour)
    
    # Create a mask with the same dimensions as the binary image, initialized to zero (black)
    mask = np.zeros_like(binary_image)
    
    # Draw the ellipse on the mask with white color (255)
    cv2.ellipse(mask, ellipse, (255), -1)
    
    # Calculate the number of pixels within the ellipse
    ellipse_area = np.sum(mask == 255)
    
    # Calculate the number of white pixels within the ellipse
    covered_pixels = np.sum((binary_image == 255) & (mask == 255))
    
    # Calculate the percentage of covered white pixels within the ellipse
    if ellipse_area == 0:
        print("area was 0")
        return ellipse_goodness  # Avoid division by zero if the ellipse area is somehow zero
    
    #percentage of covered pixels to number of pixels under area
    ellipse_goodness[0] = covered_pixels / ellipse_area
    
    #skew of the ellipse (less skewed is better?) - may not need this
    axes_lengths = ellipse[1]  # This is a tuple (minor_axis_length, major_axis_length)
    major_axis_length = axes_lengths[1]
    minor_axis_length = axes_lengths[0]
    ellipse_goodness[2] = min(ellipse[1][1]/ellipse[1][0], ellipse[1][0]/ellipse[1][1])
    
    return ellipse_goodness


def _show_tracking_windows(frame, gl_image=None, status_text=None):
    if status_text:
        cv2.putText(frame, status_text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(frame, status_text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow(eye_window_name("Tracking"), frame)

    if GL_SPHERE_AVAILABLE and gl_image is not None:
        blended = cv2.addWeighted(frame, 0.6, gl_image, 0.4, 0)
        cv2.imshow(eye_window_name("Tracker + Sphere"), blended)


# Process frames for pupil detection
def process_frames(thresholded_image_strict, thresholded_image_medium, thresholded_image_relaxed, frame, gray_frame, darkest_point, debug_mode_on, render_cv_window):
    global ray_lines
    global max_rays
    global prev_model_center_avg
    global max_observed_distance

    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    dilated_image = cv2.dilate(thresholded_image_medium, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    reduced_contours = filter_contours_by_area_and_return_largest(contours, 1000, 3)

    final_rotated_rect = ((0,0),(0,0),0)

    image_array = [thresholded_image_relaxed, thresholded_image_medium, thresholded_image_strict] #holds images
    name_array = ["relaxed", "medium", "strict"] #for naming windows
    final_image = image_array[0] #holds return array
    final_contours = [] #holds final contours
    ellipse_reduced_contours = [] #holds an array of the best contour points from the fitting process
    goodness = 0 #goodness value for best ellipse
    best_array = 0 
    kernel_size = 5  # Size of the kernel (5x5)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gray_copy1 = gray_frame.copy()
    gray_copy2 = gray_frame.copy()
    gray_copy3 = gray_frame.copy()
    gray_copies = [gray_copy1, gray_copy2, gray_copy3]
    final_goodness = 0
    
    #iterate through binary images and see which fits the ellipse best
    for i in range(1,4):
        # Dilate the binary image
        dilated_image = cv2.dilate(image_array[i-1], kernel, iterations=2)#medium
        
        # Find contours
        contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create an empty image to draw contours
        contour_img2 = np.zeros_like(dilated_image)
        reduced_contours = filter_contours_by_area_and_return_largest(contours, 1000, 3)

        #initialize variables
        center_x, center_y = None, None

        if len(reduced_contours) > 0 and len(reduced_contours[0]) > 5:
            current_goodness = check_ellipse_goodness(dilated_image, reduced_contours[0], debug_mode_on)
            ellipse = cv2.fitEllipse(reduced_contours[0])
            center_x, center_y = map(int, ellipse[0]) 
            if debug_mode_on: #show contours 
                cv2.imshow(name_array[i-1] + " threshold", gray_copies[i-1])
                
            #in total pixels, first element is pixel total, next is ratio
            total_pixels = check_contour_pixels(reduced_contours[0], dilated_image.shape, debug_mode_on)                 
            
            cv2.ellipse(gray_copies[i-1], ellipse, (255, 0, 0), 2)  # Draw with specified color and thickness of 2
            font = cv2.FONT_HERSHEY_SIMPLEX  # Font type
            
            final_goodness = current_goodness[0]*total_pixels[0]*total_pixels[0]*total_pixels[1]

        if final_goodness > 0 and final_goodness > goodness: 
            goodness = final_goodness
            ellipse_reduced_contours = total_pixels[2]
            best_image = image_array[i-1]
            final_contours = reduced_contours
            final_image = dilated_image

    test_frame = frame.copy()
    
    final_contours = [optimize_contours_by_angle(final_contours, gray_frame)]
    
    final_rotated_rect = None

    if final_contours and not isinstance(final_contours[0], list) and len(final_contours[0] > 5):
        ellipse = cv2.fitEllipse(final_contours[0])
        final_rotated_rect = ellipse

        # Store the new ray in the list
        ray_lines.append(final_rotated_rect)
        # **Prune rays if list exceeds max_rays**
        if len(ray_lines) > max_rays:
            num_to_remove = len(ray_lines) - max_rays
            ray_lines = ray_lines[num_to_remove:]  # Keep only the last `max_rays` elements

    global sphere_center_locked_2d, locked_model_center_avg, prev_model_center_avg

    model_center_average = (320,240)

    model_center = compute_average_intersection(frame, ray_lines, 5, 1500, 5)

    if not sphere_center_locked_2d:
        # Normal behavior: keep updating running average while unlocked
        if model_center is not None:
            model_center_average = update_and_average_point(model_centers, model_center, 200)
        else:
            model_center_average = prev_model_center_avg

        # If we got something sensible, remember it as the last good value
        if model_center_average[0] != 0:
            prev_model_center_avg = model_center_average
            locked_model_center_avg = model_center_average
    else:
        # Once locked, always use the frozen center
        model_center_average = locked_model_center_avg

    
    # Example safety check — still refresh the window so it does not look frozen.
    if center_x is None or center_y is None or model_center_average[0] is None or model_center_average[1] is None:
        sphere_radius = int(max_observed_distance) if max_observed_distance else 202
        if model_center_average[0] is not None:
            cv2.circle(frame, model_center_average, sphere_radius, (255, 50, 50), 2)
            cv2.circle(frame, model_center_average, 8, (255, 255, 0), -1)
        _show_tracking_windows(frame, status_text="No pupil detected")
        return final_rotated_rect

    # Calculate the distance only if model_centers has at least 100 values
    if len(model_centers) >= 100 and center_x is not None:
        distance = math.sqrt((center_x - model_center_average[0]) ** 2 + (center_y - model_center_average[1]) ** 2)
        if distance > max_observed_distance:
            max_observed_distance = distance
            
    max_observed_distance = 202

    # Draw reference lines/ellipses
    cv2.circle(frame, model_center_average, int(max_observed_distance), (255, 50, 50), 2)  # Draw eye sphere (circle)
    cv2.circle(frame, model_center_average, 8, (255, 255, 0), -1)  # Draw eye center



    if final_rotated_rect is not None and center_x is not None and center_y is not None:
        cv2.line(frame, model_center_average, (center_x, center_y), (255, 150, 50), 2)  # # Draw line from eye center to ellipse center
        
    cv2.ellipse(frame, final_rotated_rect, (20, 255, 255), 2) #draw final ellipse on image

    # Calculate the extended endpoint of gaze line
    if final_rotated_rect is not None and center_x is not None and center_y is not None:
        # Compute the vector from model_center_average to center_x, center_y
        dx = center_x - model_center_average[0]
        dy = center_y - model_center_average[1]

        # Scale the vector by 1.2x
        extended_x = int(model_center_average[0] + 2 * dx)
        extended_y = int(model_center_average[1] + 2 * dy)

        # Draw the extended gaze line
        cv2.line(frame, (center_x, center_y), (extended_x, extended_y), (200, 255, 0), 3) 




    if render_cv_window:
        cv2.imshow("Best Thresholded Image Contours on Frame", frame)


    gl_image = None
    if GL_SPHERE_AVAILABLE:
        gl_image = gl_sphere.update_sphere_rotation(center_x, center_y, model_center_average[0], model_center_average[1])
    #cv2.circle(frame, (center_x, center_y), 22, (255, 255, 0), -1)  # Draw intersection center

    # Call the function
    center, direction = compute_gaze_vector(center_x, center_y, model_center_average[0], model_center_average[1])

    if center is not None and direction is not None:
        origin_text = f"Origin: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})"
        dir_text    = f"Direction: ({direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f})"

        # Set bottom-left corner for drawing text
        text_origin = (12, frame.shape[0] - 38)  # 40 pixels from bottom
        text_dir    = (12, frame.shape[0] - 13)  # 15 pixels from bottom
        text_origin2 = (10, frame.shape[0] - 40)  # 40 pixels from bottom
        text_dir2    = (10, frame.shape[0] - 15)  # 15 pixels from bottom

        # Draw shadow text on the frame
        cv2.putText(frame, origin_text, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(frame, dir_text, text_dir, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        # Draw text on the frame
        cv2.putText(frame, origin_text, text_origin2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, dir_text, text_dir2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if center is not None and direction is not None:
        update_eye_gaze_output(current_eye_id, center, direction)

    _show_tracking_windows(frame, gl_image=gl_image)

    return final_rotated_rect

def update_and_average_point(point_list, new_point, N):
    """
    Adds a new point to the list, keeps only the last N points, 
    and returns the average of those points.
    
    Parameters:
    - point_list: Global list storing past points [(x1, y1), (x2, y2), ...]
    - new_point: Tuple (x, y) representing the new point to add.
    - N: Maximum number of points to keep in the list.
    
    Returns:
    - (avg_x, avg_y): The average point as a tuple of integers.
    - None if the list is empty.
    """
    point_list.append(new_point)  # Add new point

    if len(point_list) > N:
        point_list.pop(0)  # Remove the oldest point to maintain size N

    if not point_list:
        return None  # No points available

    avg_x = int(np.mean([p[0] for p in point_list]))
    avg_y = int(np.mean([p[1] for p in point_list]))

    return (avg_x, avg_y)

def draw_orthogonal_ray(image, ellipse, length=100, color=(0, 255, 0), thickness=1):
    """
    Draws a ray passing through the center of an ellipse orthogonally to its major axis.
    
    Parameters:
    - image: The OpenCV image to draw on.
    - ellipse: A tuple ((cx, cy), (major_axis, minor_axis), angle) representing the fitted ellipse.
    - length: Length of the ray to draw on each side of the ellipse center.
    - color: Color of the line in BGR format (default: green).
    - thickness: Thickness of the line (default: 2).
    """

    (cx, cy), (major_axis, minor_axis), angle = ellipse
    
    # Convert angle to radians
    angle_rad = np.deg2rad(angle)
    
    # Compute the normal vector at the ellipse center (perpendicular to surface)
    normal_dx = (minor_axis / 2) * np.cos(angle_rad)  # Minor axis component
    normal_dy = (minor_axis / 2) * np.sin(angle_rad)

    # Compute start and end points of the orthogonal ray
    pt1 = (int(cx - length * normal_dx / (minor_axis / 2)), int(cy - length * normal_dy / (minor_axis / 2)))
    pt2 = (int(cx + length * normal_dx / (minor_axis / 2)), int(cy + length * normal_dy / (minor_axis / 2)))

    # Draw the ray
    cv2.line(image, pt1, pt2, color, thickness)

    return image

def compute_average_intersection(frame, ray_lines, N, M, spacing):
    """
    Selects N random lines from the list, highlights them in red on the frame,
    computes their intersections, stores them, and prunes stored intersections when exceeding M.

    Parameters:
    - frame: The OpenCV frame to draw on.
    - ray_lines: List of ellipse tuples ((cx, cy), (major_axis, minor_axis), angle).
    - N: Number of random lines to select for intersection calculation.
    - M: Maximum number of stored intersections before pruning.

    Returns:
    - (avg_x, avg_y): Average intersection point of selected lines.
    """
    global stored_intersections

    if len(ray_lines) < 2 or N < 2:
        return (0, 0)  # Need at least 2 lines to find intersections

    # Get frame dimensions dynamically
    height, width = frame.shape[:2]

    # Select N unique random lines
    selected_lines = random.sample(ray_lines, min(N, len(ray_lines)))

    intersections = []

    # Highlight selected rays in red
    #for ray in selected_lines:
    #    draw_orthogonal_ray(frame, ray, color=(0, 0, 255), thickness=2)  # Red lines

    # Compute intersections for each pair of selected lines
    for i in range(len(selected_lines) - 1):
        line1 = selected_lines[i]
        line2 = selected_lines[i + 1]

        angle1 = line1[2]  # Extract angle from ellipse tuple
        angle2 = line2[2]  # Extract angle from ellipse tuple

        if abs(angle1 - angle2) >= 2:  # Ensure lines differ by at least 2 degrees
            intersection = find_line_intersection(line1, line2)
            
            # Ensure the intersection is within the frame bounds before adding
            if intersection and (0 <= intersection[0] < width) and (0 <= intersection[1] < height):
                intersections.append(intersection)
                stored_intersections.append(intersection)  # Store valid intersections
        #else:
        #    print(f"Skipped intersection: Angle difference too small ({abs(angle1 - angle2):.2f}°)")

    # Prune intersections if stored list exceeds M
    if len(stored_intersections) > M:
        stored_intersections = prune_intersections(stored_intersections, M)

    # Draw all stored intersections on the frame
    #for pt in stored_intersections:
    #    cv2.circle(frame, pt, 3, (255, 255, 255), -1)  # White dot for every past intersection

    if not intersections:
        return None  # No valid intersections found

    # Compute the average intersection point
    avg_x = np.mean([pt[0] for pt in stored_intersections])
    avg_y = np.mean([pt[1] for pt in stored_intersections])


    return (int(avg_x), int(avg_y))

#Removes the oldest intersections to ensure only the last M intersections remain.
def prune_intersections(intersections, maximum_intersections):

    if len(intersections) <= maximum_intersections:
        return intersections  # No need to prune if within the limit

    # Keep only the last M intersections
    pruned_intersections = intersections[-maximum_intersections:]

    return pruned_intersections

def rotation_from_a_to_b(a, b):
    """
    Compute rotation matrix R such that R @ a = b
    using Rodrigues' rotation formula.
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    v = np.cross(a, b)
    c = np.dot(a, b)

    if np.linalg.norm(v) < 1e-6:
        # Vectors are parallel or nearly so
        if c > 0:
            return np.eye(3, dtype=np.float32)
        else:
            # 180-degree flip: choose any axis orthogonal to a
            axis = np.array([1.0, 0.0, 0.0])
            if abs(a[0]) > 0.9:
                axis = np.array([0.0, 1.0, 0.0])
            v = np.cross(a, axis)
            v = v / np.linalg.norm(v)
            s = np.linalg.norm(v)
    else:
        s = np.linalg.norm(v)
        v = v / s

    # Skew-symmetric cross-product matrix
    vx, vy, vz = v
    K = np.array([
        [0,    -vz,  vy],
        [vz,    0,  -vx],
        [-vy,  vx,   0 ]
    ], dtype=np.float32)

    R = np.eye(3, dtype=np.float32) + K * s + (K @ K) * ((1 - c) / (s ** 2))
    return R

def update_gaze_circle_from_current_gaze():
    """
    Use the latest gaze vector to update the circle position on the external camera.
    Assumes we have calibrated R_gaze_to_cam that maps gaze_dir to external cam space.
    """
    global circle_x, circle_y, last_gaze_dir, calibrated
    global _smoothed_circle_u, _smoothed_circle_v

    if not calibrated:
        return

    gaze_dir = get_combined_gaze_dir()
    if gaze_dir is None:
        return

    g = R_gaze_to_cam @ gaze_dir

    # Avoid weird cases where gaze points behind the camera
    if g[2] <= 1e-6:
        return

    # Simple pinhole projection onto 2D
    u = float(np.clip(EXT_CX + EXT_FX * (g[0] / g[2]), 0, EXT_WIDTH - 1))
    v = float(np.clip(EXT_CY - EXT_FY * (g[1] / g[2]), 0, EXT_HEIGHT - 1))

    if _smoothed_circle_u is None:
        _smoothed_circle_u, _smoothed_circle_v = u, v
    else:
        alpha = GAZE_CIRCLE_SMOOTH_ALPHA
        _smoothed_circle_u = (1.0 - alpha) * _smoothed_circle_u + alpha * u
        _smoothed_circle_v = (1.0 - alpha) * _smoothed_circle_v + alpha * v

    circle_x = int(round(_smoothed_circle_u))
    circle_y = int(round(_smoothed_circle_v))

def find_line_intersection(ellipse1, ellipse2):
    """
    Computes the intersection of two lines that are orthogonal to the surface of given ellipses.
    
    Parameters:
    - ellipse1, ellipse2: Ellipse tuples ((cx, cy), (major_axis, minor_axis), angle).
    
    Returns:
    - (x, y): Intersection point of the two lines, or None if parallel.
    """

    (cx1, cy1), (_, minor_axis1), angle1 = ellipse1
    (cx2, cy2), (_, minor_axis2), angle2 = ellipse2

    # Convert angles to radians
    angle1_rad = np.deg2rad(angle1)
    angle2_rad = np.deg2rad(angle2)

    # Compute direction vectors for the two lines
    dx1, dy1 = (minor_axis1 / 2) * np.cos(angle1_rad), (minor_axis1 / 2) * np.sin(angle1_rad)
    dx2, dy2 = (minor_axis2 / 2) * np.cos(angle2_rad), (minor_axis2 / 2) * np.sin(angle2_rad)

    # Line equations in parametric form:
    # (x1, y1) + t1 * (dx1, dy1) = (x2, y2) + t2 * (dx2, dy2)
    A = np.array([[dx1, -dx2], [dy1, -dy2]])
    B = np.array([cx2 - cx1, cy2 - cy1])

    # Solve for t1, t2 using linear algebra (if the determinant is nonzero)
    if np.linalg.det(A) == 0:
        return None  # Lines are parallel and do not intersect

    t1, t2 = np.linalg.solve(A, B)

    # Compute intersection point
    intersection_x = cx1 + t1 * dx1
    intersection_y = cy1 + t1 * dy1

    return (int(intersection_x), int(intersection_y))

def compute_gaze_vector(x, y, center_x, center_y, screen_width=640, screen_height=480):
    """Compute 3D gaze direction from pupil and sphere center screen coordinates.
    Returns:
        sphere_center (np.ndarray): 3D position of the sphere center in world space
        gaze_direction (np.ndarray): Normalized 3D direction vector from sphere center
    """

    # Get viewport dimensions
    viewport_width = screen_width
    viewport_height = screen_height

    # Define camera and projection settings
    fov_y_deg = 45.0
    aspect_ratio = viewport_width / viewport_height
    far_clip = 100.0

    # Camera position is fixed at z = 3
    camera_position = np.array([0.0, 0.0, 3.0])

    # Compute size of far plane in world units
    fov_y_rad = np.radians(fov_y_deg)
    half_height_far = np.tan(fov_y_rad / 2) * far_clip
    half_width_far = half_height_far * aspect_ratio

    # Convert screen (x, y) to normalized device coordinates [-1, 1]
    ndc_x = (2.0 * x) / viewport_width - 1.0
    ndc_y = 1.0 - (2.0 * y) / viewport_height

    # Project pupil center to far plane coordinates in world space
    far_x = ndc_x * half_width_far
    far_y = ndc_y * half_height_far
    far_z = camera_position[2] - far_clip
    far_point = np.array([far_x, far_y, far_z])

    # Compute ray direction from camera to far plane point
    ray_origin = camera_position
    ray_direction = far_point - camera_position
    ray_direction /= np.linalg.norm(ray_direction)
    ray_direction = -ray_direction

    # Sphere radius and center offset
    inner_radius = 1.0 / 1.05
    sphere_offset_x = (center_x / screen_width) * 2.0 - 1.0
    sphere_offset_y = 1.0 - (center_y / screen_height) * 2.0
    sphere_center = np.array([sphere_offset_x * 1.5, sphere_offset_y * 1.5, 0.0])

    # Compute intersection with sphere
    origin = ray_origin
    direction = -ray_direction
    L = origin - sphere_center

    a = np.dot(direction, direction)
    b = 2 * np.dot(direction, L)
    c = np.dot(L, L) - inner_radius**2

    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        # Compute the closest point to the sphere (tangent point approximation)
        t = -np.dot(direction, L) / np.dot(direction, direction)
        intersection_point = origin + t * direction
        intersection_local = intersection_point - sphere_center
        target_direction = intersection_local / np.linalg.norm(intersection_local)
    else:
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)

        t = None
        if t1 > 0 and t2 > 0:
            t = min(t1, t2)
        elif t1 > 0:
            t = t1
        elif t2 > 0:
            t = t2
        if t is None:
            return None, None

    # Final intersection point
    intersection_point = origin + t * direction
    intersection_local = intersection_point - sphere_center
    target_direction = intersection_local / np.linalg.norm(intersection_local)

    sqrt_disc = np.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)

    t = None
    if t1 > 0 and t2 > 0:
        t = min(t1, t2)
    elif t1 > 0:
        t = t1
    elif t2 > 0:
        t = t2
    if t is None:
        return None, None

    # Final intersection point
    intersection_point = origin + t * direction

    # Convert to local space relative to sphere center
    intersection_local = intersection_point - sphere_center
    target_direction = intersection_local / np.linalg.norm(intersection_local)

    # Local green ring direction
    circle_local_center = np.array([0.0, 0.0, inner_radius])
    circle_local_center /= np.linalg.norm(circle_local_center)

    # Compute rotation to align local +Z to target
    rotation_axis = np.cross(circle_local_center, target_direction)
    rotation_axis_norm = np.linalg.norm(rotation_axis)
    if rotation_axis_norm < 1e-6:
        return sphere_center, circle_local_center

    rotation_axis /= rotation_axis_norm
    dot = np.dot(circle_local_center, target_direction)
    dot = np.clip(dot, -1.0, 1.0)
    angle_rad = np.arccos(dot)

    # Rotation matrix from axis-angle
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    t_ = 1 - c
    x_, y_, z_ = rotation_axis

    rotation_matrix = np.array([
        [t_*x_*x_ + c, t_*x_*y_ - s*z_, t_*x_*z_ + s*y_],
        [t_*x_*y_ + s*z_, t_*y_*y_ + c, t_*y_*z_ - s*x_],
        [t_*x_*z_ - s*y_, t_*y_*z_ + s*x_, t_*z_*z_ + c]
    ])

     # Rotate +Z vector to get gaze direction
    gaze_local = np.array([0.0, 0.0, inner_radius])
    gaze_rotated = rotation_matrix @ gaze_local
    gaze_rotated /= np.linalg.norm(gaze_rotated)

    global last_sphere_center, last_gaze_dir, calibrated_sphere_center
    last_sphere_center = sphere_center.copy()
    last_gaze_dir = gaze_rotated.copy()
    eye_tracking_states[current_eye_id]["last_sphere_center"] = last_sphere_center
    eye_tracking_states[current_eye_id]["last_gaze_dir"] = last_gaze_dir

    if calibrated_sphere_center is not None:
        sphere_center_out = calibrated_sphere_center
    else:
        sphere_center_out = sphere_center

    return sphere_center_out, gaze_rotated

def on_mouse_frame_with_rays(event, x, y, flags, param):
    """Left-click on an eye tracking window to lock that eye's sphere center."""
    global sphere_center_locked_2d, locked_model_center_avg, prev_model_center_avg
    global calibrated_sphere_center, calibrated, last_sphere_center

    eye_id = param if param in EYE_IDS else current_eye_id
    load_eye_tracking_state(eye_id)

    if event == cv2.EVENT_LBUTTONDOWN:
        locked_model_center_avg = (x, y)
        prev_model_center_avg = locked_model_center_avg
        sphere_center_locked_2d = True

        if last_sphere_center is not None:
            calibrated_sphere_center = last_sphere_center.copy()
            calibrated = True
            print(f"[{eye_id}] Manual sphere center set at 2D:", locked_model_center_avg)
            print(f"[{eye_id}] Fixed eye origin (sphere center 3D):", calibrated_sphere_center)
        else:
            print(f"[{eye_id}] Manual 2D center set at:", locked_model_center_avg,
                  "but no 3D sphere center available yet.")

    save_eye_tracking_state(eye_id)


def calibrate_gaze_to_external(active_eyes=EYE_IDS):
    global calibrated, R_gaze_to_cam

    refresh_combined_gaze(active_eyes)
    combined = get_combined_gaze_dir()
    if combined is None:
        print("Calibration failed: no gaze vector available yet.")
        return

    forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    R_gaze_to_cam = rotation_from_a_to_b(combined, forward)

    for eye_id in active_eyes:
        state = eye_tracking_states[eye_id]
        if state["last_sphere_center"] is None:
            continue

        load_eye_tracking_state(eye_id)
        calibrated_sphere_center = state["last_sphere_center"].copy()
        sphere_center_locked_2d = True
        locked_model_center_avg = prev_model_center_avg
        save_eye_tracking_state(eye_id)
        print(f"[{eye_id}] 2D sphere center locked at:", locked_model_center_avg)

    calibrated = True
    refresh_combined_gaze(active_eyes)
    write_gaze_vector_file()
    print("Calibration complete (combined gaze aligned to front camera forward).")





def process_frame(frame, eye_id="left", flip_vertical=False, flip_horizontal=False):
    if flip_vertical:
        frame = cv2.flip(frame, 0)
    if flip_horizontal:
        frame = cv2.flip(frame, 1)

    load_eye_tracking_state(eye_id)
    try:
        frame = crop_to_aspect_ratio(frame)
        darkest_point = get_darkest_area(frame)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        darkest_pixel_value = gray_frame[darkest_point[1], darkest_point[0]]

        thresholded_image_strict = apply_binary_threshold(gray_frame, darkest_pixel_value, 5)
        thresholded_image_strict = mask_outside_square(thresholded_image_strict, darkest_point, 250)

        thresholded_image_medium = apply_binary_threshold(gray_frame, darkest_pixel_value, 15)
        thresholded_image_medium = mask_outside_square(thresholded_image_medium, darkest_point, 250)

        thresholded_image_relaxed = apply_binary_threshold(gray_frame, darkest_pixel_value, 25)
        thresholded_image_relaxed = mask_outside_square(thresholded_image_relaxed, darkest_point, 250)

        return process_frames(
            thresholded_image_strict,
            thresholded_image_medium,
            thresholded_image_relaxed,
            frame,
            gray_frame,
            darkest_point,
            False,
            False,
        )
    finally:
        save_eye_tracking_state(eye_id)
