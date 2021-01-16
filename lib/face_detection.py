import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

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