from flask import Flask, render_template
import base64
import cv2
import imageio
import requests

app = Flask(__name__)
camera = cv2.VideoCapture(0)

server_url = "http://localhost:5000"
configure_url = server_url + "/configure"

@app.route('/configure', methods = ['POST'])
def configure():
    source_image = imageio.imread('images/ryan.png')
    source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
    source, source_buffer = cv2.imencode('.jpg', source_image)

    _, frame = camera.read()
    frame, frame_buffer = cv2.imencode('.jpg', frame)

    source_encoded = base64.b64encode(source_buffer)
    frame_encoded = base64.b64encode(frame_buffer)

    data = {
      "source": source_encoded,
      "frame": frame_encoded,
    }

    r = requests.post(configure_url, data=data)
    
    if r.status_code != 204:
      print("error")
      
    return ('', 204)


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(port=3000)