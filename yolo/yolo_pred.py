import sys
from matplotlib.pyplot import grid
sys.path.append('.')

import tensorflow as tf
from voc.voc_datagen import *
from utils import draw_box, load_image

import os
import cv2


S = 4
C = 20
img_size = 224
grid_size = img_size // S

def post_progress(output):
    
    for i in range(S):
        for j in range(S):
            grid_vector = output[0, i, j]
            tf.print(grid_vector)
            # [ob_exist, x, y, w, h, ..., C]
            if grid_vector[0] > 0:
                # exist object
                ob_x = grid_vector[1] + i
                ob_y = grid_vector[2] + j
                ob_w = grid_vector[3]
                ob_h = grid_vector[4]
                # tf.print(ob_x, ob_y, ob_w, ob_h)
                ob_x = ob_x * grid_size
                ob_y = ob_y * grid_size
                ob_w = ob_w * grid_size
                ob_h = ob_h * grid_size
                # tf.print(ob_x, ob_y, ob_w, ob_h)
                xmin = ob_x - ob_w / 2
                ymin = ob_y - ob_h / 2
                xmax = ob_x + ob_w / 2
                ymax = ob_y + ob_h / 2
                # tf.print(xmin, ymin, xmax, ymax)
                cls_vector = grid_vector[5:]
                # for cls in cls_vector:
                #     tf.print(cls)




if __name__ == '__main__':
    
    test_img = os.path.join(voc_image_path, '2008_000002.jpg')
    img = load_image(test_img)
    input = tf.expand_dims(img, axis=0)
    model = tf.keras.models.load_model("model/yolo.h5")
    output = model(input)

    post_progress(output)
    