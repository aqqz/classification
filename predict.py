import tensorflow as tf
from utils import load_image
import os
from keras.preprocessing.image import image
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == '__main__':
    
    test_image = '/home/taozhi/datasets/VOCdevkit/VOC2012/JPEGImages/2008_000026.jpg'
    # 测试h5模型
    net = tf.keras.models.load_model("model/voc.h5")
    img = load_image(test_image)
    
    image.save_img('test.pgm', img, mode='gray')
    input = tf.expand_dims(img, axis=0)
    output = net.predict(input)
    print(output)




    # 测试tflite模型
    # model = tf.lite.Interpreter(model_path='model/voc.tflite')
    # model.allocate_tensors()

    # input_details = model.get_input_details()[0]
    # output_details = model.get_output_details()[0]

    # img = load_image(test_image)
    # if input_details["dtype"] == np.uint8:
    #     input_scale, input_zero_point = input_details["quantization"]
    #     img = img / input_scale + input_zero_point

    # input = np.expand_dims(img, axis=0).astype(input_details["dtype"])
    # model.set_tensor(input_details['index'], input)
    # model.invoke()
    # output_data = model.get_tensor(output_details['index'])
    # print(output_data)