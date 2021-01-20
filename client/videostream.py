import threading
from imutils.video import VideoStream
import time

camera_frame = None
camera_frame_lock = threading.Lock()

def get_next_frame(frame_rate=24):
    global camera_frame, camera_frame_lock
    sleep_time = 1/frame_rate
    camera = VideoStream(src=0).start()
    while True:
        new_frame = camera.read()

        with camera_frame_lock:
            camera_frame = new_frame

        time.sleep(sleep_time)

"""
Blocking call returns copy of frame
"""
def get_frame():
    global camera_frame, camera_frame_lock
    frame = None
    with camera_frame_lock:
        if camera_frame is not None:
            frame = camera_frame.copy()
    return frame