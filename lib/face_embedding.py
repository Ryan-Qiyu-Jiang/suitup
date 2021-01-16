import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

embedding_model_path = '../lib/embedding_model/openface_nn4.small2.v1.t7'
embedder = cv2.dnn.readNetFromTorch(embedding_model_path)

def embed_face(img):
    face_blob = cv2.dnn.blobFromImage(img, 1.0 / 255, (96, 96), 
        (0, 0, 0), swapRB=True, crop=False)
    embedder.setInput(face_blob)
    vec = embedder.forward()
    return vec