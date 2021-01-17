from flask import Flask, render_template, Response
import cv2

from transform import transform_init, transform, crop_img
import imageio
from skimage import img_as_ubyte

app = Flask(__name__)

camera = cv2.VideoCapture(0)

source_image = imageio.imread('images/hannah.png')
source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
success, frame = camera.read()
driving_image = frame

source_tensor, kp_source, kp_driving_initial = transform_init(source_image, driving_image)

def gen_tranformed_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            frame = cv2.flip(crop_img(frame), 1)
            transformed_frame = transform(kp_source, kp_driving_initial, frame, source_tensor)
            frame_ubytes = img_as_ubyte(transformed_frame)
            ret, buffer = cv2.imencode('.jpg', frame_ubytes)
            stream_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + stream_bytes + b'\r\n')  # concat frame one by one and show result

def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            frame = cv2.flip(crop_img(frame), 1)
            ret, buffer = cv2.imencode('.jpg', frame)
            stream_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + stream_bytes + b'\r\n')

@app.route('/configure', methods = ['POST'])
def configure():
    global kp_driving_initial
    _, driving_image = camera.read()
    _, _, kp_driving_initial = transform_init(source_image, driving_image)
    return ('', 204)

@app.route('/transformed_feed')
def transformed_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_tranformed_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
