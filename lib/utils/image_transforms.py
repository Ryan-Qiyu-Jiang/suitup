import cv2
import numpy as np


def center_crop(img, crop_size=None):
    h, w = img.shape[:2]
    if crop_size is None:
        crop_size = min(w, h)
    s = crop_size//2
    s_x = max(w//2-s, 0)
    e_x = min(w//2+s, w)
    s_y = max(h//2-s, 0)
    e_y = min(h//2+s, h)
    cropped = img[s_y:e_y, s_x:e_x]
    return cropped


def scale_img(img, dim):
    if img.shape[0] > 256 and img.shape[1] > 256:
        return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)


def scale_crop(img):
    h, w = img.shape[:2]
    s = 256/min(h,w)
    img = scale_img(img, (int(round(w*s)), int(round(h*s))))
    img = center_crop(img, 256)
    return img


def correct_dim(img):
    return (img.shape and img.shape[0] == 256 and img.shape[1] == 256)
