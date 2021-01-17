import random
import string
from collections import defaultdict

from flask import Flask, render_template, Response
import cv2

from transform import transform_init, transform, crop_img
import imageio
from skimage import img_as_ubyte

app = Flask(__name__)

users = defaultdict(lambda x : None)

def generate_uid():
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for i in range(8))


def gen_transformed_frames(frame, uid):
    # When you read initially, it is flipped
    frame = cv2.flip(crop_img(frame), 1)

    user_info = users[uid]
    if not user_info:
      return None

    # Use the stuff we have stored for the user
    transformed_frame = transform(
        user_info["kp_source"],
        user_info["kp_driving_initial"],
        frame,
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
    source_image = request.json["source"]
    frame = request.json["frame"]

    source_tensor, kp_source, kp_driving_initial = transform_init(source_image, frame)

    uid = generate_uid()
    while users[uid] != None:
      uid = generate_uid()

    users[uid] = {
      "kp_source": kp_source,
      "kp_driving_initial": kp_driving_initial,
      "source_tensor": source_tensor,
    }

    # TODO: Hook into verification here

    return (uid, 200)


@app.route('/transform')
def transform():
    # Configure will take a source image and a frame
    uid = request.json["uid"]
    frame = request.json["frame"]

    return (gen_transformed_frames(frame, uid), 200)
