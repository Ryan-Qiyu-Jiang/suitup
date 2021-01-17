import cv2
import numpy as np


def center_crop(img, crop_size=None):
    w = img.shape[0]
    h = img.shape[1]
    if crop_size is None:
        crop_size = min(w, h)
    startx = w//2-(crop_size//2)
    starty = h//2-(crop_size//2)
    cropped = img[starty:starty+crop_size, startx:startx+crop_size]
    return cropped


def scale_img(img, dim):
    if img.shape[0] > 256 and img.shape[1] > 256:
        return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)


def scale_crop(img):
    h,w = img.shape[:2]
    s = 256/min(h,w)
    img = scale_img(img, (int(w*s), int(h*s)))
    img = center_crop(img, 256)
    return img


def correct_dim(img):
    return (img.shape and img.shape[0] == 256 and img.shape[1] == 256)
