import tensorflow as tf
from voc.voc_datagen import *
from utils import load_data
import numpy as np
import time
import os
from keras.preprocessing.image import image

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def representative_data_gen():
    for input_value in tf.data.Dataset.from_tensor_slices(test_images).batch(1).take(100):
        # Model has only one input so each data point has one element.
        yield [input_value]

def lite_convert(model_path, quantization="none", save_path="model/model.tflite"):

    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantization=='int8':
        print("quantizing model by int8...\n")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.target_spec.supported_types = [tf.int8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    print("converting model...\n")
    tflite_model = converter.convert()  

    with open(save_path, "wb") as f:
        f.write(tflite_model)


def evaluate_tflite(model_path, test_images, test_labels):
    print("evaluating tflite model...\n")
    start = time.time()
    # 创建解释器
    interpreter = tf.lite.Interpreter(model_path)
    # 分配张量
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # tf.print(input_details)
    # tf.print(output_details)

    count = 0
    accuracy = tf.keras.metrics.Accuracy()

    if input_details["dtype"] == np.uint8:
        input_scale, input_zero_point = input_details["quantization"]
        test_images = test_images / input_scale + input_zero_point

    for test_image in test_images:
        input_data = np.expand_dims(test_image, axis=0).astype(input_details['dtype'])
        # print(input_data)
        test_label = test_labels[count]
        # print(test_label)
        interpreter.set_tensor(input_details['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details['index'])[0]
        print(output_data)
        # 计算分类精度
        accuracy.update_state(tf.argmax(test_label), tf.argmax(output_data))
        count += 1

    end = time.time()
    print("test tflite on: {} examples, accuracy: {}, time: {}".format(
        count, accuracy.result().numpy(), end-start
    ))

    # 清零
    count = 0
    accuracy.reset_states()

def save_samples(x_test, y_test, len, mode='gray'):
    print(f"generate {len} samples for quantize.")
    x_quant = x_test[:len]
    y_quant = y_test[:len]
    count=0
    if os.path.exists('samples') == False:
        os.mkdir('samples')
    for i in x_quant:
        image.save_img('samples/' + str(count) + '_' + str(y_quant[count]) + '.pgm', i, mode=mode)
        count += 1

    print("saved samples in samples/")


if __name__ == '__main__':

    test_img_paths, test_img_labels = generate_paths_labels_list(cls="person", \
        voc_label_path=voc_label_path, voc_image_path=voc_image_path, mode="val")
    
    test_images, test_labels = load_data(test_img_paths, test_img_labels)

    lite_convert('model/voc.h5', quantization="none", save_path="model/voc.tflite")

    save_samples(test_images, test_labels, len=100, mode="gray")
    
    evaluate_tflite(model_path="model/voc.tflite", test_images=test_images, test_labels=test_labels)

    

    

    
    

    
    