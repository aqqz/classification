import sys
sys.path.append('.')

import tensorflow as tf
from data_gen import *
from utils import draw_box, load_image

import os
import cv2



def post_progress(img_w, img_h, output):
    ob_infos = []
    for i in range(S):
        for j in range(S):
            grid_vector = output[0, i, j]
            # tf.print(grid_vector)
            # [ob_exist, x, y, w, h, ..., C]
            if grid_vector[0] > 0.4:
                scale_x = img_w / img_size
                scale_y = img_h / img_size
                normal_x = grid_vector[1]
                normal_y = grid_vector[2]
                normal_w = grid_vector[3]
                normal_h = grid_vector[4]
                # tf.print(normal_x, normal_y, normal_w, normal_h)
                # exist object
                ob_x = (normal_x + i) * grid_size
                ob_y = (normal_y + j) * grid_size
                ob_w = normal_w * img_size
                ob_h = normal_h * img_size
                # tf.print(ob_x, ob_y, ob_w, ob_h)
                ob_x *= scale_x
                ob_y *= scale_y
                ob_w *= scale_x
                ob_h *= scale_y
                # tf.print(ob_x, ob_y, ob_w, ob_h)
                xmin = int(ob_x - ob_w / 2)
                ymin = int(ob_y - ob_h / 2) 
                xmax = int(ob_x + ob_w / 2) 
                ymax = int(ob_y + ob_h / 2)
                # tf.print(xmin, ymin, xmax, ymax)
                cls_vector = grid_vector[5:]
                cls = tf.argmax(cls_vector).numpy()
                # tf.print(cls)
                ob_infos.append([cls, xmin, ymin, xmax, ymax])
    # print(ob_infos)
    return ob_infos
    




if __name__ == '__main__':
    
    test_img = os.path.join(voc_image_path, '2007_000032.jpg')
    img = load_image(test_img)
    input = tf.expand_dims(img, axis=0)
    model = tf.keras.models.load_model("model/yolo.h5")
    output = model(input)

    img = cv2.imread(test_img)
    img_w, img_h = img.shape[1], img.shape[0]
    ob_infos = post_progress(img_w, img_h, output)
    draw_box(test_img, ob_infos)