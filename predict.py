import tensorflow as tf
from train import load_image
import time
import tensorflow_hub as hub
import os
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if __name__ == '__main__':

    test_image = '/home/taozhi/archive/test/117196494.jpg'
    data_root = '/home/taozhi/archive/train' # 训练数据根目录
    class_names = os.listdir(data_root)

    start = time.time()
    net = tf.keras.models.load_model('model/model.h5', custom_objects={'KerasLayer': hub.KerasLayer})
    input = tf.expand_dims(load_image(test_image), 0)
    output = tf.argmax(net.predict(input), axis=1)[0]
    end = time.time()

    print(f"class: {output}, result: {class_names[output]}, time: {end-start}")
    plt.imshow(load_image(test_image))
    plt.show()