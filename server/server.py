import sys
sys.path.append('../')

from lib.utils.image_transforms import correct_dim
import time
from skimage import img_as_ubyte
import imageio
from facial_verification import verify_same_face, face_distance
from face_transform import transform_init, transform as transform_image, crop_img
import cv2
from flask import Flask, render_template, Response, request
import numpy as np
import base64
from collections import defaultdict
import string
import random


app = Flask(__name__)

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
    _, buffer = cv2.imencode('.jpg', frame_ubytes)
    stream_bytes = buffer.tobytes()
    return stream_bytes


@app.route('/configure', methods=['POST'])
def configure():
    # Configure will take a source image and a frame
    encoded_source = request.form["source"]
    encoded_frame = request.form["frame"]

    source_as_np = np.fromstring(base64.b64decode(encoded_source), np.uint8)
    frame_as_np = np.fromstring(base64.b64decode(encoded_frame), np.uint8)

    decoded_source = cv2.imdecode(source_as_np, cv2.IMREAD_COLOR)

    decoded_frame = cv2.imdecode(frame_as_np, cv2.IMREAD_COLOR)
    assert correct_dim(decoded_source)
    assert correct_dim(decoded_frame)

    distance = face_distance(decoded_source, decoded_frame)
    print('distance', distance)
    if distance > 0.3:
        # TODO: Figure out not-same-face behavior
        return ("#", 200)

    source_tensor, kp_source, kp_driving_initial = transform_init(
        decoded_source, decoded_frame)

    uid = generate_uid()
    while users[uid] != None:
        uid = generate_uid()

    users[uid] = {
        "kp_source": kp_source,
        "kp_driving_initial": kp_driving_initial,
        "source_tensor": source_tensor,
    }

    return (uid, 200)


@app.route('/transform', methods=['POST'])
def transform():
    # Configure will take a source image and a frame
    uid = request.form["uid"]
    encoded_frame = request.form["frame"]

    frame_as_np = np.fromstring(base64.b64decode(encoded_frame), np.uint8)
    decoded_frame = cv2.imdecode(frame_as_np, cv2.IMREAD_COLOR)

    res = (gen_transformed_frames(decoded_frame, uid), 200)
    return res


if __name__ == '__main__':
    app.run()
