import sys
sys.path.append('.')

import tensorflow as tf
from voc.voc_datagen import *
from utils import draw_box

import os
import cv2


if __name__ == '__main__':
    
    test_img = os.path.join(voc_image_path, '')