import sys
sys.path.append('../')

from lib.utils.image_transforms import scale_crop
import requests
import imageio
import cv2
import base64
from flask import Flask, render_template, Response, redirect, url_for, request
import numpy as np
from lib.utils.profiling import frame_count, timer

import threading

from videostream import camera_frame_lock, camera_frame, get_next_frame, get_frame
from lib.face_detection import get_facial_roi, get_face_location, get_source_frame

app = Flask(__name__)
# camera = VideoStream(src=0).start() #cv2.VideoCapture(0)

server_url = 'http://127.0.0.1:5000'
# server_url = 'http://9aaa3ade2048.ngrok.io'
# server_url = "http://34.221.75.41:5000"
configure_url = server_url + "/configure"
transform_url = server_url + "/transform"

uid = None
NULL_UID = '#'
jpeg_quality = 95

source_url = 'static/images/jack3.jpg'
source_image = imageio.imread(source_url)
source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
# source_image = cv2.flip(scale_crop(source_image), 1)
transform_pos = (0,0)
transform_size = (256,256)

def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame 
        frame = get_frame()

        if frame is None:
            continue
        else:
            frame = cv2.flip(scale_crop(frame), 1)
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            stream_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + stream_bytes + b'\r\n')


@app.route('/configure', methods=['POST'])
def configure():
    global transform_pos, transform_size
    source = source_image.copy()
    rois = get_facial_roi(source)
    s_x, s_y, e_x, e_y = get_face_location(source, rois)
    source, pos = get_source_frame(source, s_x, s_y, e_x, e_y)
    transform_pos = pos
    transform_size = source.shape[:2]
    source = scale_crop(source)

    _, source_buffer = cv2.imencode('.jpg', source)

    frame = get_frame()
    frame = cv2.flip(scale_crop(frame), 1)
    _, frame_buffer = cv2.imencode('.jpg', frame)

    source_encoded = base64.b64encode(source_buffer)
    frame_encoded = base64.b64encode(frame_buffer)

    data = {
        "source": source_encoded,
        "frame": frame_encoded,
    }

    r = requests.post(configure_url, data=data)

    if r.status_code != 200:
        print("error")
        return ('', 204)

    global uid
    uid = r.text

    if uid == NULL_UID:
      return redirect(url_for('mismatch'))

    return redirect(url_for('index'))


def gen_transformed_frames():
    while True:
        # Capture frame-by-frame
        frame = get_frame()

        if frame is None:
            break
        else:
            frame = cv2.flip(scale_crop(frame), 1)
            _, frame_buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            frame_encoded = base64.b64encode(frame_buffer)
            data = {
                "uid": uid,
                "frame": frame_encoded
            }
            # frame_count()
            # timer('')
            r = requests.post(transform_url, data=data)
            # print(timer('server_processing'))
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + r.content + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/transform')
def transform():
    if uid and uid is not NULL_UID:
        return Response(gen_transformed_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/upload-image", methods=["POST"])
def upload_image():
    if request.method == "POST":
        if request.files:
            image = request.files["image"]

            global source_image
            source_image = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR)
            source_image = cv2.flip(scale_crop(source_image), 1)

    return redirect(url_for('index'))


@app.route('/')
def index():
    """Video streaming home page."""
    if uid and uid is not NULL_UID :
        return render_template('transformed.html', source_image=source_url, 
            transform_pos=transform_pos, transform_size=transform_size)
    return render_template('index.html') 


@app.route("/upload", methods=["GET", "POST"])
def upload():
    return render_template("upload.html")

@app.route("/mismatch", methods=["GET", "POST"])
def mismatch():
    return render_template("mismatch.html")

if __name__ == '__main__':
    t = threading.Thread(target=get_next_frame, args=(24,))
    t.daemon = True
    t.start()
    app.run(port=3000)
