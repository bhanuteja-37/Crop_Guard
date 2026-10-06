from ultralytics import YOLO
import cv2
import json
import os
from collections import deque

from camera.detector.risk_engine import BirdRiskEngine
from camera.detector.zone_detector import get_zone
from camera.detector.decision_engine import choose_response

from hardware.deterrent import (
    activate_deterrent,
    deactivate_deterrent
)

from utils.logger import save_alert


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


# ============================================================
# CONFIGURATION FILES
# ============================================================

CONFIG_FILE = os.path.join(
    BASE_DIR,
    "config",
    "crop_profiles.json"
)

CURRENT_CONFIG_FILE = os.path.join(
    BASE_DIR,
    "config",
    "current_config.json"
)


# ============================================================
# LOAD CROP PROFILES
# ============================================================

try:

    with open(CONFIG_FILE, "r") as file:
        crop_profiles = json.load(file)

except FileNotFoundError:

    print("ERROR: crop_profiles.json not found.")
    print(f"Expected location: {CONFIG_FILE}")
    exit()


# ============================================================
# LOAD CURRENT CONFIGURATION
# ============================================================

try:

    with open(CURRENT_CONFIG_FILE, "r") as file:
        current_config = json.load(file)

except FileNotFoundError:

    print("ERROR: current_config.json not found.")
    print(f"Expected location: {CURRENT_CONFIG_FILE}")
    exit()


# ============================================================
# CURRENT CROP AND GROWTH STAGE
# ============================================================

crop = current_config.get(
    "crop",
    "rice"
)

growth_stage = current_config.get(
    "growth_stage",
    "early"
)


print(
    f"Current configuration | "
    f"Crop: {crop} | "
    f"Stage: {growth_stage}"
)


# ============================================================
# VALIDATE CROP
# ============================================================

if crop not in crop_profiles:

    print(
        f"ERROR: Invalid crop '{crop}'."
    )

    print(
        "Available crops:",
        list(crop_profiles.keys())
    )

    exit()


# ============================================================
# VALIDATE GROWTH STAGE
# ============================================================

if growth_stage not in crop_profiles[crop]["stages"]:

    print(
        f"ERROR: Invalid growth stage "
        f"'{growth_stage}'."
    )

    print(
        "Available stages:",
        crop_profiles[crop]["stages"]
    )

    exit()


print(
    f"CropGuard configuration loaded | "
    f"Crop: {crop_profiles[crop]['name']} | "
    f"Stage: {growth_stage}"
)


# ============================================================
# LOAD YOLO
# ============================================================

print("Loading YOLO model...")

model = YOLO("yolo11n.pt")

print("YOLO model loaded successfully!")


# ============================================================
# RISK ENGINE
# ============================================================

risk_engine = BirdRiskEngine(
    crop=crop,
    growth_stage=growth_stage
)


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(0)


if not camera.isOpened():

    print("Camera could not be opened.")
    exit()


print("CropGuard AI camera started.")
print("Press Q to exit.")


# ============================================================
# DETECTION STABILITY SETTINGS
# ============================================================

REQUIRED_CONSECUTIVE_DETECTIONS = 3

bird_detection_count = 0

confidence_history = deque(
    maxlen=5
)


# ============================================================
# RESPONSE STATE
# ============================================================

last_response = None

last_logged_zone = None

last_logged_risk = None


# ============================================================
# MAIN CAMERA LOOP
# ============================================================

while True:

    ret, frame = camera.read()


    if not ret:

        print("Could not read camera frame.")

        break


    # --------------------------------------------------------
    # YOLO DETECTION
    # --------------------------------------------------------

    results = model(frame)


    bird_detected = False

    highest_confidence = 0.0

    zone = None


    # --------------------------------------------------------
    # FIND BIRD
    # --------------------------------------------------------

    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])

            confidence = float(box.conf[0])

            class_name = model.names[class_id]


            if class_name == "bird":

                bird_detected = True


                if confidence > highest_confidence:

                    highest_confidence = confidence


                    x1, y1, x2, y2 = box.xyxy[0]


                    x_center = float(
                        (x1 + x2) / 2
                    )


                    frame_width = frame.shape[1]


                    zone = get_zone(
                        x_center,
                        frame_width
                    )


    # ========================================================
    # BIRD DETECTION COUNT
    # ========================================================

    if bird_detected:

        bird_detection_count += 1

        confidence_history.append(
            highest_confidence
        )

    else:

        bird_detection_count = 0

        confidence_history.clear()

        last_response = None

        last_logged_zone = None

        last_logged_risk = None


    # ========================================================
    # CHECK REPEATED DETECTION
    # ========================================================

    repeated_detection = (
        bird_detection_count
        >= REQUIRED_CONSECUTIVE_DETECTIONS
    )


    # ========================================================
    # CALCULATE STABLE CONFIDENCE
    # ========================================================

    if confidence_history:

        stable_confidence = sum(
            confidence_history
        ) / len(confidence_history)

    else:

        stable_confidence = 0.0


    # ========================================================
    # RISK + DECISION
    # ========================================================

    if bird_detected:

        score, risk = risk_engine.calculate_risk(

            confidence=stable_confidence,

            repeated_detection=repeated_detection,

            zone=zone
        )


        response = choose_response(risk)


        print(
            f"Bird detected | "
            f"Confidence: "
            f"{stable_confidence:.2f} | "
            f"Frames: "
            f"{bird_detection_count} | "
            f"Repeated: "
            f"{repeated_detection} | "
            f"Zone: "
            f"{zone} | "
            f"Risk: "
            f"{risk} | "
            f"Score: "
            f"{score} | "
            f"Response: "
            f"{response}"
        )


        # ====================================================
        # DETERRENT CONTROL
        # ====================================================

        if response == "ACTIVATE_DETERRENT":

            if last_response != response:

                activate_deterrent(zone)


        elif response == "MONITOR":

            if last_response == "ACTIVATE_DETERRENT":

                deactivate_deterrent(zone)


        elif response == "NO_ACTION":

            if last_response == "ACTIVATE_DETERRENT":

                deactivate_deterrent(zone)


        last_response = response


        # ====================================================
        # EVENT LOGGING
        # ====================================================

        event_changed = (

            risk != last_logged_risk

            or zone != last_logged_zone

        )


        if event_changed:

            save_alert(

                crop=crop,

                stage=growth_stage,

                confidence=round(
                    stable_confidence,
                    2
                ),

                zone=zone,

                risk=risk,

                response=response

            )


            last_logged_risk = risk

            last_logged_zone = zone


    else:

        risk = "LOW"

        response = "NO_ACTION"


    # ========================================================
    # DRAW YOLO RESULTS
    # ========================================================

    annotated_frame = results[0].plot()


    frame_height, frame_width = (
        annotated_frame.shape[:2]
    )


    # ========================================================
    # DRAW ZONE LINES
    # ========================================================

    zone_width = frame_width // 3


    cv2.line(

        annotated_frame,

        (zone_width, 0),

        (zone_width, frame_height),

        (255, 255, 255),

        2

    )


    cv2.line(

        annotated_frame,

        (zone_width * 2, 0),

        (zone_width * 2, frame_height),

        (255, 255, 255),

        2

    )


    # ========================================================
    # ZONE LABELS
    # ========================================================

    cv2.putText(

        annotated_frame,

        "ZONE 1",

        (20, 40),

        cv2.FONT_HERSHEY_SIMPLEX,

        1,

        (255, 255, 255),

        2

    )


    cv2.putText(

        annotated_frame,

        "ZONE 2",

        (zone_width + 20, 40),

        cv2.FONT_HERSHEY_SIMPLEX,

        1,

        (255, 255, 255),

        2

    )


    cv2.putText(

        annotated_frame,

        "ZONE 3",

        (zone_width * 2 + 20, 40),

        cv2.FONT_HERSHEY_SIMPLEX,

        1,

        (255, 255, 255),

        2

    )


    # ========================================================
    # CROP INFORMATION
    # ========================================================

    cv2.putText(

        annotated_frame,

        f"Crop: {crop_profiles[crop]['name']}",

        (20, 75),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (255, 255, 255),

        2

    )


    cv2.putText(

        annotated_frame,

        f"Stage: {growth_stage}",

        (20, 105),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (255, 255, 255),

        2

    )


    # ========================================================
    # BIRD INFORMATION
    # ========================================================

    if bird_detected:

        cv2.putText(

            annotated_frame,

            f"Bird Zone: {zone}",

            (20, frame_height - 90),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (255, 255, 255),

            2

        )


        cv2.putText(

            annotated_frame,

            f"Confidence: {stable_confidence:.2f}",

            (20, frame_height - 60),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (255, 255, 255),

            2

        )


        cv2.putText(

            annotated_frame,

            f"Risk: {risk}",

            (20, frame_height - 30),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (255, 255, 255),

            2

        )


    # ========================================================
    # SHOW CAMERA
    # ========================================================

    cv2.imshow(

        "CropGuard - AI Bird Detection",

        annotated_frame

    )


    # ========================================================
    # EXIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

camera.release()

cv2.destroyAllWindows()

print("CropGuard AI camera stopped.")