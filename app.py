import streamlit as st
import cv2
import tempfile
import os
import time
import requests

from ultralytics import YOLO


# --------------------------------
# Page configuration
# --------------------------------

st.set_page_config(
    page_title="Wildlife Detection System",
    page_icon="🐘",
    layout="wide"
)


# --------------------------------
# Custom CSS
# --------------------------------

st.markdown("""
<style>

.main {
    background-color: #f5f8f6;
}

.title {
    font-size: 38px;
    font-weight: 700;
    color: #163b2a;
}

.subtitle {
    color: #6b7280;
    font-size: 17px;
}

.upload-box {
    padding: 30px;
    border-radius: 15px;
    background-color: white;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.08);
}

.detection-card {
    padding: 20px;
    border-radius: 12px;
    background-color: white;
    margin-bottom: 15px;
    box-shadow: 0px 3px 12px rgba(0,0,0,0.08);
}

</style>
""", unsafe_allow_html=True)


# --------------------------------
# Configuration
# --------------------------------

MODEL_PATH = "wild_animal_detector.pt"

ALERT_API = (
    "https://lvidya.pythonanywhere.com/api/sendnotification"
)

CONFIDENCE_THRESHOLD = 0.5

ALERT_COOLDOWN = 5 * 60


CLASS_NAMES = {
    0: "Elephant",
    1: "Leopard",
    2: "Tiger",
    3: "Wild-Boar"
}


# --------------------------------
# Load model
# --------------------------------

@st.cache_resource
def load_model():

    return YOLO(MODEL_PATH)


wildlife_model = load_model()


# --------------------------------
# Alert state
# --------------------------------

if "last_alert_time" not in st.session_state:
    st.session_state.last_alert_time = {}

if "alerted_tracks" not in st.session_state:
    st.session_state.alerted_tracks = set()


# --------------------------------
# Send alert
# --------------------------------

def send_alert(animal, track_id, confidence):

    payload = {
        "animal": animal,
        "track_id": track_id,
        "confidence": round(float(confidence), 2)
    }

    try:

        response = requests.post(
            ALERT_API,
            json=payload,
            timeout=10
        )

        if response.ok:

            return True

        st.warning(
            f"Alert API returned {response.status_code}"
        )

    except requests.exceptions.RequestException as e:

        st.error(f"Alert API error: {e}")

    return False


# --------------------------------
# Header
# --------------------------------

st.markdown(
    '<div class="title">🐘 Wildlife Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload a CCTV video to detect wildlife -Elephant, Leopard, Tiger, and Wild-Boar.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# --------------------------------
# Upload video
# --------------------------------

uploaded_file = st.file_uploader(
    "Upload CCTV Video",
    type=["mp4", "avi", "mov", "mkv"]
)


if uploaded_file is not None:

    st.video(uploaded_file)

    st.write(
        f"**File:** {uploaded_file.name}"
    )

    st.write(
        f"**Size:** "
        f"{uploaded_file.size / (1024 * 1024):.2f} MB"
    )


    start = st.button(
        "🔍 Start Wildlife Detection",
        type="primary"
    )


    if start:

        # --------------------------------
        # Save temporary video
        # --------------------------------

        suffix = os.path.splitext(
            uploaded_file.name
        )[1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(
                uploaded_file.read()
            )

            video_path = temp_file.name


        # --------------------------------
        # Output area
        # --------------------------------

        st.subheader(
            "🎥 Detection in progress"
        )

        video_placeholder = st.empty()
        progress_bar = st.progress(0)

        status_text = st.empty()


        # --------------------------------
        # Open video
        # --------------------------------

        cap = cv2.VideoCapture(
            video_path
        )

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        frame_width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        frame_height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )


        # --------------------------------
        # Detection results
        # --------------------------------

        detected_animals = {}

        frame_number = 0


        # --------------------------------
        # YOLO tracking
        # --------------------------------

        results = wildlife_model.track(
            source=video_path,
            conf=CONFIDENCE_THRESHOLD,
            tracker="bytetrack.yaml",
            imgsz=640,
            max_det=100,
            stream=True,
            persist=True
        )


        for result in results:

            frame_number += 1

            if total_frames > 0:

                progress = (
                    frame_number /
                    total_frames
                )

                progress_bar.progress(
                    min(progress, 1.0)
                )


            status_text.write(
                f"Processing frame "
                f"{frame_number}/{total_frames}"
            )


            # ----------------------------
            # No tracked objects
            # ----------------------------

            if result.boxes.id is None:

                continue


            track_ids = (
                result.boxes.id
                .int()
                .cpu()
                .tolist()
            )

            class_ids = (
                result.boxes.cls
                .int()
                .cpu()
                .tolist()
            )

            confidences = (
                result.boxes.conf
                .cpu()
                .tolist()
            )


            # ----------------------------
            # Process detections
            # ----------------------------

            for (
                track_id,
                class_id,
                confidence
            ) in zip(
                track_ids,
                class_ids,
                confidences
            ):

                animal = CLASS_NAMES.get(
                    class_id,
                    "Unknown"
                )

                confidence = float(
                    confidence
                )


                # ------------------------
                # Store highest confidence
                # ------------------------

                if animal not in detected_animals:

                    detected_animals[animal] = {
                        "track_ids": set(),
                        "confidence": confidence
                    }

                detected_animals[
                    animal
                ]["track_ids"].add(track_id)


                if confidence > detected_animals[
                    animal
                ]["confidence"]:

                    detected_animals[
                        animal
                    ]["confidence"] = confidence


               # ------------------------
                # Alert control
                # ------------------------

                current_time = time.time()

                # Unique key for this animal and tracking ID
                alert_key = f"{animal}_{track_id}"

                # Last alert time for this animal
                last_alert = st.session_state.last_alert_time.get(animal, 0)

                time_since_alert = current_time - last_alert

                # Debug information
                status_text.write(
                    f"Frame {frame_number} | "
                    f"{animal} | "
                    f"Track ID: {track_id} | "
                    f"Confidence: {confidence:.2f}"
                )

                # Send alert only if:
                # 1. This animal+track has not already alerted
                # 2. Cooldown for this animal has expired

                if (
                    alert_key not in st.session_state.alerted_tracks
                    and
                    time_since_alert >= ALERT_COOLDOWN
                ):

                    success = send_alert(
                        animal,
                        track_id,
                        confidence
                    )

                    if success:

                        # Remember this animal + track
                        st.session_state.alerted_tracks.add(
                            alert_key
                        )

                        # Start cooldown for this animal
                        st.session_state.last_alert_time[
                            animal
                        ] = current_time

                        st.success(
                            f"🚨 Alert sent: "
                            f"{animal} | "
                            f"Track ID: {track_id} | "
                            f"Confidence: {confidence:.2f}"
                        )


        cap.release()


        # --------------------------------
        # Finish
        # --------------------------------

        progress_bar.progress(1.0)

        status_text.success(
            "✅ Video processing completed"
        )


        # --------------------------------
        # Results
        # --------------------------------

        st.divider()

        st.subheader(
            "🦌 Detection Results"
        )


        if detected_animals:

            cols = st.columns(
                len(detected_animals)
            )


            for col, (
                animal,
                data
            ) in zip(
                cols,
                detected_animals.items()
            ):

                with col:

                    st.metric(
                        animal,
                        f"{data['confidence'] * 100:.1f}%"
                    )

                    st.caption(
                        f"Tracks detected: "
                        f"{len(data['track_ids'])}"
                    )

        else:

            st.info(
                "No wildlife was detected "
                "in this video."
            )


        # --------------------------------
        # Cleanup
        # --------------------------------

        if os.path.exists(video_path):

            os.remove(video_path)