import random
import string
from collections import defaultdict

import base64
import numpy as np

from flask import Flask, render_template, Response, request
from flask_socketio import SocketIO, send, emit
import cv2

from face_transform import transform_init, transform as transform_image, crop_img
from facial_verification import verify_same_face, face_distance

import imageio
from skimage import img_as_ubyte

app = Flask(__name__)
socketio = SocketIO(app, logger=True, engineio_logger=True)

users = defaultdict(lambda: None)

def generate_uid():
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for i in range(8))


def gen_transformed_frames(decoded_frame, uid):
    if not uid in users:
      return None
      
    user_info = users[uid]
    if not user_info:
      return None

    # Use the stuff we have stored for the user
    transformed_frame = transform_image(
        user_info["kp_source"],
        user_info["kp_driving_initial"],
        decoded_frame,
        user_info["source_tensor"],
    )

    # Process the transformed image
    frame_ubytes = img_as_ubyte(transformed_frame)
    ret, buffer = cv2.imencode('.jpg', frame_ubytes)
    stream_bytes = buffer.tobytes()
    return stream_bytes


@app.route('/configure', methods = ['POST'])
def configure():
    # Configure will take a source image and a frame
    encoded_source = request.form["source"]
    encoded_frame = request.form["frame"]

    source_as_np = np.fromstring(base64.b64decode(encoded_source), np.uint8)
    frame_as_np = np.fromstring(base64.b64decode(encoded_frame), np.uint8)

    decoded_source = cv2.imdecode(source_as_np, cv2.IMREAD_COLOR)
    # decoded_source = cv2.cvtColor(decoded_source, cv2.COLOR_BGR2RGB)
    decoded_source = cv2.flip(crop_img(decoded_source), 1)

    decoded_frame = cv2.imdecode(frame_as_np, cv2.IMREAD_COLOR)
    decoded_frame = cv2.cvtColor(decoded_frame, cv2.COLOR_BGR2RGB)
    decoded_frame = cv2.flip(crop_img(decoded_frame), 1)

    distance = face_distance(decoded_source, decoded_frame)
    print('distance', distance)
    if False and distance > 0.3:
      # TODO: Figure out not-same-face behavior
      return ("#", 200)

    source_tensor, kp_source, kp_driving_initial = transform_init(decoded_source, decoded_frame)

    uid = generate_uid()
    while users[uid] != None:
      uid = generate_uid()

    users[uid] = {
      "kp_source": kp_source,
      "kp_driving_initial": kp_driving_initial,
      "source_tensor": source_tensor,
    }

    return (uid, 200)


@app.route('/transform', methods = ['POST'])
def transform():
    # Configure will take a source image and a frame
    uid = request.form["uid"]
    encoded_frame = request.form["frame"]

    frame_as_np = np.fromstring(base64.b64decode(encoded_frame), np.uint8)
    decoded_frame = cv2.imdecode(frame_as_np, cv2.IMREAD_COLOR)
    decoded_frame = cv2.cvtColor(decoded_frame, cv2.COLOR_BGR2RGB)
    decoded_frame = cv2.flip(crop_img(decoded_frame), 1)

    return (gen_transformed_frames(decoded_frame, uid), 200)


@socketio.on('message')
def handle_message(data):
    print('received message: ' + data)

@socketio.on('data')
def handle_message(data):
    print('received data')
    uid = data["uid"]
    encoded_frame = data["frame"]

    frame_as_np = np.fromstring(base64.b64decode(encoded_frame), np.uint8)
    decoded_frame = cv2.imdecode(frame_as_np, cv2.IMREAD_COLOR)
    decoded_frame = cv2.cvtColor(decoded_frame, cv2.COLOR_BGR2RGB)
    decoded_frame = cv2.flip(crop_img(decoded_frame), 1)

    emit('transformed', gen_transformed_frames(decoded_frame, uid))


@socketio.on('connect')
def handle_connect():
    print("client connected")


@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')


if __name__ == '__main__':
    print("Server running on " + str(5000))
    socketio.run(app)
