import tensorflow as tf
from utils import load_image
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def predict(img_path, class_names):
    """
    输入分类名和一张图片，返回预测结果
    """
    net = tf.keras.models.load_model("model/model.h5")
    img = load_image(img_path)
    input = np.expand_dims(img, axis=0)
    output = net.predict(input)[0]
    res = np.argmax(output)
    print(output)
    print("result: {}, probility: {}".format(class_names[res], output[res]))


def predict_tflite(img_path):
    """
    预测tflite模型
    """
    model = tf.lite.Interpreter(model_path='model/model.tflite')
    model.allocate_tensors()

    input_details = model.get_input_details()[0]
    output_details = model.get_output_details()[0]
    
    tf.print(output_details)

    img = load_image(img_path)
    if input_details["dtype"] == np.uint8:
        input_scale, input_zero_point = input_details["quantization"]
        img = img / input_scale + input_zero_point

    input = np.expand_dims(img, axis=0).astype(input_details["dtype"])
    model.set_tensor(input_details['index'], input)
    model.invoke()
    output_data = model.get_tensor(output_details['index'])[0]
    # print(output_data)

    if output_details["dtype"] == np.uint8:
        output_scale, output_zero_point = output_details["quantization"]
        output_data = (output_data - output_zero_point) * output_scale
    print(output_data)    



if __name__ == '__main__':
    
    data_root = '/home/taozhi/datasets/ds' # 训练数据根目录
    class_names = os.listdir(data_root)
    test_image = '/home/taozhi/datasets/ds/room/img_1.jpeg'
    
    # 测试h5模型
    predict(test_image, class_names)

    # 测试tflite模型
    # predict_tflite(test_image)