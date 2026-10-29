
import cv2
import mediapipe as mp
import copy
import itertools
from tensorflow import keras
import numpy as np
import string
import os
import time
from datetime import datetime
from collections import deque, Counter

# Paths relative to this script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "model.h5")
SCREENSHOTS_DIR = os.path.join(SCRIPT_DIR, "screenshots")

# Load the saved model from file
model = keras.models.load_model(MODEL_PATH)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Classes: 1-9 then A-Z (35 classes)
alphabet = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet += list(string.ascii_uppercase)

# Gesture taxonomy in Indian Sign Language (ISL)
TWO_HANDED_SIGNS = {
    'A', 'B', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'W', 'X', 'Y', 'Z'
}
ONE_HANDED_SIGNS = {
    '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'C', 'I', 'L', 'O', 'U', 'V'
}

# Stability smoothing queue (stores last N predictions to prevent flicker)
SMOOTHING_WINDOW = 6
history_deque = deque(maxlen=SMOOTHING_WINDOW)

# Screenshot settings
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
countdown_start = None
countdown_duration = 5


def calc_landmark_list(image, landmarks):
    """Convert normalized landmarks to pixel coordinates."""
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for landmark in landmarks.landmark:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    """Normalize landmarks relative to wrist and scale to [-1, 1]."""
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates based on wrist (index 0)
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
    for index in range(len(temp_landmark_list)):
        temp_landmark_list[index][0] -= base_x
        temp_landmark_list[index][1] -= base_y

    # Flatten to 1D
    flat_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalize by maximum absolute coordinate value
    max_value = max(map(abs, flat_list))
    if max_value == 0:
        return flat_list

    return [coord / max_value for coord in flat_list]


def draw_hud(image, sign, mode, confidence, hand_labels):
    """Draw a clean top HUD overlay displaying sign status and tracking info."""
    h, w = image.shape[:2]

    # Create semi-transparent dark header banner
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)

    # Separator line under header
    cv2.line(image, (0, 80), (w, 80), (70, 70, 70), 1)

    if sign is not None:
        # Determine color badge based on sign mode
        if "Two-Handed" in mode:
            badge_color = (0, 230, 115)   # Bright green
        elif "Single-Handed" in mode:
            badge_color = (255, 180, 50)   # Cyan / sky blue
        else:
            badge_color = (0, 140, 255)    # Orange / warning for missing 2nd hand

        # Left side: Detected Sign
        cv2.putText(
            image, f"Sign: {sign}",
            (25, 45), cv2.FONT_HERSHEY_DUPLEX,
            1.2, (255, 255, 255), 2, cv2.LINE_AA
        )

        # Mode Badge
        cv2.putText(
            image, f"[{mode}]",
            (25, 70), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, badge_color, 2, cv2.LINE_AA
        )

        # Right side: Confidence & Hand Details
        conf_text = f"Confidence: {int(confidence * 100)}%"
        cv2.putText(
            image, conf_text,
            (w - 280, 38), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (220, 220, 220), 1, cv2.LINE_AA
        )

        hands_str = ", ".join(hand_labels) if hand_labels else "None"
        hands_text = f"Hands: {len(hand_labels)} ({hands_str})"
        cv2.putText(
            image, hands_text,
            (w - 280, 65), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (170, 170, 170), 1, cv2.LINE_AA
        )
    else:
        # Idle status when no hands are in frame
        cv2.putText(
            image, "Indian Sign Language Recognition (ISL)",
            (25, 38), cv2.FONT_HERSHEY_DUPLEX,
            0.75, (200, 200, 200), 1, cv2.LINE_AA
        )
        cv2.putText(
            image, "Waiting for hands... (Show 1 hand for C, 2 hands for A, B, etc.)",
            (25, 65), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (130, 130, 130), 1, cv2.LINE_AA
        )

    # Footer instructions
    footer_text = "Press 's' for 5s Screenshot  |  'q' or ESC to Exit"
    cv2.putText(
        image, footer_text,
        (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
        0.50, (180, 180, 180), 1, cv2.LINE_AA
    )


def main():
    global countdown_start

    cap = cv2.VideoCapture(0)

    # Configure MediaPipe Hands for up to 2 hands
    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip image horizontally for natural selfie view
            image = cv2.flip(image, 1)
            h, w = image.shape[:2]

            # Convert color space for MediaPipe
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            image.flags.writeable = True

            frame_sign = None
            frame_mode = "No Hands"
            frame_conf = 0.0
            hand_labels = []

            if results.multi_hand_landmarks:
                num_hands = len(results.multi_hand_landmarks)
                preprocessed_list = []
                wrist_positions = []

                # Extract landmark features and render skeletal lines for each hand
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness
                ):
                    hand_label = handedness.classification[0].label
                    hand_labels.append(hand_label)

                    raw_points = calc_landmark_list(image, hand_landmarks)
                    preprocessed = pre_process_landmark(raw_points)
                    preprocessed_list.append(preprocessed)

                    # Save wrist point for hand-local badge
                    wrist_pt = (
                        int(hand_landmarks.landmark[0].x * w),
                        int(hand_landmarks.landmark[0].y * h)
                    )
                    wrist_positions.append((wrist_pt, hand_label))

                    # Draw hand landmarks and connections
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                # Fast vectorized batch inference
                batch_tensor = np.array(preprocessed_list, dtype=np.float32)
                raw_predictions = model(batch_tensor, training=False).numpy()

                per_hand_labels = []
                per_hand_confs = []
                for p in raw_predictions:
                    c_idx = int(np.argmax(p))
                    c_conf = float(p[c_idx])
                    per_hand_labels.append(alphabet[c_idx])
                    per_hand_confs.append(c_conf)

                # ==========================================
                # JOINT HAND FUSION LOGIC
                # ==========================================
                if num_hands == 2:
                    p0 = raw_predictions[0]
                    p1 = raw_predictions[1]
                    lbl0, conf0 = per_hand_labels[0], per_hand_confs[0]
                    lbl1, conf1 = per_hand_labels[1], per_hand_confs[1]

                    # Average combined probabilities across both hands
                    combined_prob = (p0 + p1) / 2.0
                    comb_idx = int(np.argmax(combined_prob))
                    comb_lbl = alphabet[comb_idx]
                    comb_conf = float(combined_prob[comb_idx])

                    if lbl0 == lbl1:
                        # Both hands agree on the gesture
                        frame_sign = lbl0
                        frame_conf = (conf0 + conf1) / 2.0
                        if frame_sign in TWO_HANDED_SIGNS:
                            frame_mode = "Two-Handed Sign"
                        else:
                            frame_mode = "Single-Handed Sign (Dual)"
                    elif comb_lbl in TWO_HANDED_SIGNS and comb_conf >= 0.35:
                        # Joint distribution clearly identifies a 2-handed sign (e.g., A, B)
                        frame_sign = comb_lbl
                        frame_conf = comb_conf
                        frame_mode = "Two-Handed Sign"
                    else:
                        # Hands differ: favor the hand with the highest individual confidence
                        if conf0 >= conf1:
                            frame_sign = lbl0
                            frame_conf = conf0
                        else:
                            frame_sign = lbl1
                            frame_conf = conf1
                        frame_mode = "Two-Handed Sign" if frame_sign in TWO_HANDED_SIGNS else "Single-Handed Sign"

                elif num_hands == 1:
                    lbl0 = per_hand_labels[0]
                    conf0 = per_hand_confs[0]
                    frame_sign = lbl0
                    frame_conf = conf0

                    if frame_sign in ONE_HANDED_SIGNS:
                        frame_mode = "Single-Handed Sign"
                    else:
                        frame_mode = f"Needs 2nd Hand for {frame_sign}"

                # Draw local badge above each wrist
                for i, ((wx, wy), h_label) in enumerate(wrist_positions):
                    local_text = f"{h_label}: {per_hand_labels[i]} ({int(per_hand_confs[i]*100)}%)"
                    tx = max(10, min(wx - 40, w - 160))
                    ty = max(95, wy - 20)
                    cv2.putText(
                        image, local_text,
                        (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 255), 2, cv2.LINE_AA
                    )

            # ==========================================
            # TEMPORAL SMOOTHING BUFFER
            # ==========================================
            if frame_sign is not None:
                history_deque.append((frame_sign, frame_mode, frame_conf))
                # Majority vote for stable sign
                sign_counts = Counter([item[0] for item in history_deque])
                stable_sign = sign_counts.most_common(1)[0][0]

                # Match mode and average confidence for the stable sign
                matching_items = [item for item in history_deque if item[0] == stable_sign]
                stable_mode = matching_items[-1][1]
                stable_conf = float(np.mean([item[2] for item in matching_items]))
            else:
                history_deque.clear()
                stable_sign = None
                stable_mode = "No Hands"
                stable_conf = 0.0

            # Draw Header HUD and status
            draw_hud(image, stable_sign, stable_mode, stable_conf, hand_labels)

            # Screenshot countdown handling
            if countdown_start is not None:
                remaining = countdown_duration - int(time.time() - countdown_start)
                if remaining > 0:
                    cd_text = f"Screenshot in: {remaining}s"
                    cv2.putText(
                        image, cd_text,
                        (25, 125), cv2.FONT_HERSHEY_DUPLEX,
                        1.0, (0, 255, 255), 2, cv2.LINE_AA
                    )
                else:
                    filename = datetime.now().strftime("ISL_output_%Y%m%d_%H%M%S.png")
                    filepath = os.path.join(SCREENSHOTS_DIR, filename)
                    if cv2.imwrite(filepath, image):
                        print("Screenshot saved:", os.path.abspath(filepath))
                    else:
                        print("Screenshot could not be saved.")
                    countdown_start = None

            # Render live window
            cv2.imshow('Indian sign language detector', image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and countdown_start is None:
                countdown_start = time.time()
                print("Screenshot countdown started. Get ready!")
            elif key == ord('q') or key == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
