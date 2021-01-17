import sys
sys.path.append('../')

import numpy as np
import cv2
from lib.face_embedding import embed_face
from lib.face_detection import get_facial_roi, get_face_location

def get_face(rois, image):
    start_x, start_y, end_x, end_y = get_face_location(image, rois)
    max_face = image[start_y:end_y, start_x:end_x]

    return max_face


def euclidian_distance(vec1, vec2):
    return np.sum((vec1-vec2)**2)**0.5


def cosine_distance(vec1, vec2):
    vec1 = np.squeeze(np.asarray(vec1))
    vec2 = np.squeeze(np.asarray(vec2))
    return 1-(np.dot(vec1, vec2)/(np.linalg.norm(vec1) * np.linalg.norm(vec2)))


def face_distance(source_image, driving_image):
    source_rois = get_facial_roi(source_image)
    driving_rois = get_facial_roi(driving_image)
    source_face = get_face(source_rois, source_image)
    driving_face = get_face(driving_rois, driving_image)

    assert source_face is not None
    assert driving_face is not None

    source_embedding = embed_face(source_face)
    driving_embedding = embed_face(driving_face)
    # e_distance = euclidian_distance(source_embedding, driving_embedding)
    c_distance = cosine_distance(source_embedding, driving_embedding)
    # print('cosine distance', c_distance, 'euclidian_distance', e_distance)
    return c_distance


def verify_same_face(source_image, driving_image, threshold=0.3):
    return face_distance(source_image, driving_image) < threshold


if __name__ == "__main__":
    import imageio
    import cv2

    hannah_path = 'images/hannah.png'
    jack_path = 'images/jack.png'
    ryan_path = 'images/ryan.png'
    ryan1_path = 'images/ryan1.png'
    ryan2_path = 'images/ryan2.png'
    hannah_img = cv2.imread(hannah_path)
    jack_img = cv2.imread(jack_path)
    ryan_img = cv2.imread(ryan_path)
    ryan1_img = cv2.imread(ryan1_path)
    ryan2_img = cv2.imread(ryan2_path)

    print('hannah, jack')
    verify_same_face(hannah_img, jack_img)
    print('jack, jack')
    verify_same_face(jack_img, jack_img)
    print('ryan, jack')
    verify_same_face(ryan_img, jack_img)
    print('ryan, hannah')
    verify_same_face(ryan_img, hannah_img)
    print('ryan, ryan')
    verify_same_face(ryan_img, ryan_img)
    verify_same_face(ryan_img, ryan1_img)
    verify_same_face(ryan_img, ryan2_img)
