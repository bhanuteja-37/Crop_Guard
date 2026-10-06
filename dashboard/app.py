import streamlit as st
import csv
import os
import json


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

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

LOG_FILE = os.path.join(
    BASE_DIR,
    "logs",
    "detections.csv"
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CropGuard",
    page_icon="🌾",
    layout="wide"
)


# ============================================================
# LOAD CROP PROFILES
# ============================================================

try:

    with open(CONFIG_FILE, "r") as file:
        crop_profiles = json.load(file)

except FileNotFoundError:

    st.error("crop_profiles.json not found.")
    st.stop()


# ============================================================
# LOAD CURRENT CONFIGURATION
# ============================================================

try:

    with open(CURRENT_CONFIG_FILE, "r") as file:
        current_config = json.load(file)

except (FileNotFoundError, json.JSONDecodeError):

    current_config = {
        "crop": "rice",
        "growth_stage": "early"
    }


# ============================================================
# TITLE
# ============================================================

st.title("🌾 CropGuard")

st.subheader(
    "AI-Assisted Multi-Crop Bird Protection System"
)


# ============================================================
# CROP CONFIGURATION
# ============================================================

st.divider()

st.header("🌱 Crop Configuration")


crop_keys = list(crop_profiles.keys())

crop_names = [
    crop_profiles[crop]["name"]
    for crop in crop_keys
]


# Find current crop
current_crop = current_config.get(
    "crop",
    "rice"
)

if current_crop in crop_keys:

    default_crop_index = crop_keys.index(
        current_crop
    )

else:

    default_crop_index = 0


col1, col2 = st.columns(2)


# ============================================================
# CROP
# ============================================================

with col1:

    selected_crop = st.selectbox(
        "Select Crop",
        crop_keys,
        index=default_crop_index,
        format_func=lambda x: crop_profiles[x]["name"]
    )


# ============================================================
# GROWTH STAGE
# ============================================================

available_stages = crop_profiles[
    selected_crop
]["stages"]


current_stage = current_config.get(
    "growth_stage",
    available_stages[0]
)


if current_stage in available_stages:

    default_stage_index = available_stages.index(
        current_stage
    )

else:

    default_stage_index = 0


with col2:

    selected_stage = st.selectbox(
        "Growth Stage",
        available_stages,
        index=default_stage_index
    )


# ============================================================
# SAVE CONFIGURATION
# ============================================================

if st.button(
    "💾 Save Configuration",
    type="primary"
):

    new_config = {
        "crop": selected_crop,
        "growth_stage": selected_stage
    }

    with open(
        CURRENT_CONFIG_FILE,
        "w"
    ) as file:

        json.dump(
            new_config,
            file,
            indent=4
        )

    st.success(
        f"Configuration saved: "
        f"{crop_profiles[selected_crop]['name']} "
        f"- {selected_stage}"
    )


# ============================================================
# CURRENT CONFIGURATION
# ============================================================

st.info(
    f"Current Configuration → "
    f"Crop: {crop_profiles[selected_crop]['name']} | "
    f"Stage: {selected_stage}"
)


# ============================================================
# LOAD DETECTION LOG
# ============================================================

detections = []


if os.path.exists(LOG_FILE):

    try:

        with open(
            LOG_FILE,
            "r",
            newline=""
        ) as file:

            reader = csv.DictReader(file)

            detections = list(reader)

    except Exception as error:

        st.error(
            f"Could not read detection log: {error}"
        )


# ============================================================
# DASHBOARD METRICS
# ============================================================

total_detections = len(detections)


if detections:

    latest = detections[-1]

    current_risk = latest.get(
        "Risk",
        "LOW"
    )

    current_zone = latest.get(
        "Zone",
        "None"
    )

    current_response = latest.get(
        "Response",
        "NO_ACTION"
    )

    latest_confidence = latest.get(
        "Confidence",
        "0"
    )

else:

    current_risk = "LOW"
    current_zone = "None"
    current_response = "NO_ACTION"
    latest_confidence = "0"


# ============================================================
# LIVE MONITORING
# ============================================================

st.divider()

st.header("📊 Live Monitoring")


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Bird Events",
        total_detections
    )


with col2:

    st.metric(
        "Current Risk",
        current_risk
    )


with col3:

    st.metric(
        "Affected Zone",
        current_zone
    )


with col4:

    st.metric(
        "Response",
        current_response
    )


# ============================================================
# LATEST DETECTION
# ============================================================

st.divider()

st.header("🐦 Latest Detection")


if detections:

    col1, col2, col3 = st.columns(3)

    with col1:

        st.write(
            f"**Confidence:** "
            f"{latest_confidence}"
        )

    with col2:

        st.write(
            f"**Zone:** "
            f"{current_zone}"
        )

    with col3:

        st.write(
            f"**Risk:** "
            f"{current_risk}"
        )

else:

    st.info(
        "No bird detections recorded yet."
    )


# ============================================================
# DETECTION HISTORY
# ============================================================

st.divider()

st.header("📋 Detection History")


if detections:

    st.dataframe(
        detections,
        width="stretch"
    )

else:

    st.info(
        "Detection history is empty."
    )


# ============================================================
# SYSTEM INFORMATION
# ============================================================

st.divider()

st.header("⚙️ System Information")

st.write("**AI Model:** YOLO11n")

st.write("**Detection:** Bird detection")

st.write(
    "**Risk Engine:** Confidence + repeated detection + crop + growth stage"
)

st.write("**Zone System:** 3 zones")

st.write(
    "**Decision Engine:** LOW / MEDIUM / HIGH"
)

st.write(
    "**Deterrent:** Non-harmful simulated response"
)

st.write(
    "**Storage:** CSV detection logs"
)