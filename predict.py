import tensorflow as tf
from train import load_image
import time
import os
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if __name__ == '__main__':

    test_image = 'test/dog.jpeg'
    data_root = '/home/taozhi/datasets/face2' # 训练数据根目录
    class_names = os.listdir(data_root)

    net = tf.keras.models.load_model('model/model.h5')
    input = tf.expand_dims(load_image(test_image), 0)
    start = time.time()
    res = net.predict(input)
    output = tf.argmax(res, axis=1)[0]
    end = time.time()
    print(res)
    print(f"class: {output}, result: {class_names[output]}, time: {end-start}")
    plt.imshow(load_image(test_image))
    plt.show()