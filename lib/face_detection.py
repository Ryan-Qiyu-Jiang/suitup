import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
from lib.utils.image_transforms import scale_crop

proto_path = '../lib/detection_model/deploy.prototxt'
model_path = '../lib/detection_model/res10_300x300_ssd_iter_140000.caffemodel'
detector = cv2.dnn.readNetFromCaffe(proto_path, model_path)

"""
Get regions of interest where a face might be.
Returns detections ( 0, 0, roi, 2)
"""
def get_facial_roi(img):
    img = imutils.resize(img, width=600)
    h, w = img.shape[:2]
    img_blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)
    detector.setInput(img_blob)
    detections = detector.forward()
    return detections

def get_face_location(image, rois):
    h, w = image.shape[:2]
    max_conf = 0
    s_x, s_y, e_x, e_y = rois[0, 0, 0, 3:7] * np.array([w, h, w, h])
    for i in range(rois.shape[2]):
        confidence = rois[0, 0, i, 2]
        if max_conf < confidence:
            max_conf = confidence
            box = rois[0, 0, i, 3:7] * np.array([w, h, w, h])
            start_x, start_y, end_x, end_y = box.astype("int")
            fH, fW = (end_y-start_y, end_x-start_x)
            if fW < 20 or fH < 20:
                continue
            s_x, s_y, e_x, e_y = start_x, start_y, end_x, end_y
    return (s_x, s_y, e_x, e_y)

def get_source_frame(image, start_x, start_y, end_x, end_y):
    center_x = (end_x+start_x)/2
    center_y = (end_y-start_y)/2
    fH, fW = (end_y-start_y, end_x-start_x)
    crop_size = (fH+fW)/2
    s = crop_size/2
    face = image[center_y-s:center_y+s, center_x-s:center_x+s]
    return scale_crop(face)
