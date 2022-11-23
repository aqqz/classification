import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def save_gray(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(img_path, img)


def load_image(image_path):
    """
    TensorFlow读取数据，返回image_tensor [0, 1]
    """
    raw = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(raw, channels=3)
    img = tf.image.resize(img, [224, 224])
    # img = tf.image.crop_to_bounding_box(img, 10, 50, 224, 224)
    img = tf.cast(img, tf.float32)    
    img = img / 255
    return img


if __name__ == '__main__':

    img_path = '/home/taozhi/datasets/ds/room/img_1.jpeg'
    
    img = np.array(load_image(img_path))
    
    plt.imsave('test2.jpg', img)
    