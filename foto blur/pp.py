from flask import Flask, Response, render_template, jsonify, request
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe import Image as mpImage
import threading
import os
import time
import numpy as np

app = Flask(__name__, template_folder="templates")

# --- Hand detection setup (new Tasks API) ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
hand_landmarker = vision.HandLandmarker.create_from_options(options)


def finger_up(tip, pip_idx, landmarks):
    return landmarks[tip].y < landmarks[pip_idx].y


def is_peace(landmarks):
    index_up = finger_up(8, 6, landmarks)
    middle_up = finger_up(12, 10, landmarks)
    ring_up = finger_up(16, 14, landmarks)
    pinky_up = finger_up(20, 18, landmarks)
    return index_up and middle_up and not ring_up and not pinky_up


# --- Camera streaming ---
camera = None
camera_lock = threading.Lock()
current_camera_index = 0


def list_cameras():
    available = []
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(current_camera_index)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return camera


def generate_frames():
    while True:
        with camera_lock:
            cap = get_camera()
            success, frame = cap.read()

        if not success:
            continue

        frame = cv2.flip(frame, 1)

        # Convert to MediaPipe Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mpImage(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = hand_landmarker.detect(mp_image)

        peace_detected = False
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                if is_peace(hand_landmarks):
                    peace_detected = True
                    break

        # Apply blur
        if peace_detected:
            frame = cv2.GaussianBlur(frame, (61, 61), 0)
            cv2.putText(frame, "BLUR!", (frame.shape[1] // 2 - 70, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/cameras")
def cameras():
    indices = list_cameras()
    return jsonify({"cameras": [{"id": i, "name": f"Camera {i}"} for i in indices], "current": current_camera_index})


@app.route("/switch_camera", methods=["POST"])
def switch_camera():
    global camera, current_camera_index
    data = request.get_json()
    new_index = int(data.get("camera_id", 0))
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
        current_camera_index = new_index
    return jsonify({"success": True, "camera_id": new_index})


if __name__ == "__main__":
    app.run(debug=False, port=5000, threaded=True)