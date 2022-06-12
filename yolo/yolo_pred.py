import sys
sys.path.append('.')

import tensorflow as tf
from data_gen import *
from utils import draw_box, load_image

import os
import cv2
    


if __name__ == '__main__':
    
    test_img = os.path.join(voc_image_path, '2008_000008.jpg')

    img = load_image(test_img)
    input = tf.expand_dims(img, axis=0)
    model = tf.keras.models.load_model("model/yolo.h5")
    output = model(input)

    img = cv2.imread(test_img)
    img_w, img_h = img.shape[1], img.shape[0]
    ob_infos = post_progress(img_w, img_h, output)
    draw_box(test_img, ob_infos)