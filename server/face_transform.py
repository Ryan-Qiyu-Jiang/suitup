import sys
sys.path.append('../')

from fom_model import load_checkpoints
import warnings
from lib.utils.image_transforms import correct_dim
from fom_model import normalize_kp
from IPython.display import HTML
from skimage.transform import resize
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import imageio
import cv2
from tqdm import tqdm
import torch
from skimage import img_as_ubyte


warnings.filterwarnings("ignore")

fps = 8
predictions = []
relative = True
adapt_movement_scale = True
cpu = True

generator, kp_detector = load_checkpoints(config_path='config/vox-256.yaml',
                                          checkpoint_path='checkpoints/vox-cpk.pth.tar')


def crop_img(img):
    w = img.shape[0]
    h = img.shape[1]
    crop_size = min(w, h)
    startx = w//2-(crop_size//2)
    starty = h//2-(crop_size//2)
    cropped = img[starty:starty+crop_size, startx:startx+crop_size]
    return cropped


def transform_init(source_image, driving_image):
    assert correct_dim(source_image)
    assert correct_dim(driving_image)

    with torch.no_grad():
        source = torch.tensor(source_image[np.newaxis].astype(
            np.float32)).permute(0, 3, 1, 2)/255
        if not cpu:
            source = source.cuda()
        driving = torch.tensor(driving_image[np.newaxis].astype(
            np.float32)).permute(0, 3, 1, 2)/255
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving)

    return source, kp_source, kp_driving_initial


def transform(kp_source, kp_driving_initial, driving_frame, source):
    with torch.no_grad():
        assert correct_dim(driving_frame)
        driving_frame = torch.tensor(driving_frame[np.newaxis].astype(
            np.float32)).permute(0, 3, 1, 2)/255
        if not cpu:
            driving_frame = driving_frame.cuda()
        kp_driving = kp_detector(driving_frame)
        kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                               kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                               use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
        out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
        return np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
