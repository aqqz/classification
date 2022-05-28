import tensorflow as tf
from voc.voc_datagen import *
from utils import load_data
import numpy as np
import time


def lite_convert(model_path, quantization="none", save_path="model/model.tflite", image_list=[], count=100):

    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantization=='none':
        # save sample for quantize aware training
        pass
    # https://www.tensorflow.org/lite/performance/post_training_integer_quant#%E4%BD%BF%E7%94%A8%E6%B5%AE%E7%82%B9%E5%9B%9E%E9%80%80%E9%87%8F%E5%8C%96%E8%BF%9B%E8%A1%8C%E8%BD%AC%E6%8D%A2
    elif quantization=='int8':
        print("quantizing model by int8...\n")
        def representative_dataset():
            for data in tf.data.Dataset.from_tensor_slices(image_list).batch(1).take(count):
                yield [tf.dtypes.cast(data, tf.float32)]
        
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
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
    
    count = 0
    accuracy = tf.keras.metrics.Accuracy()
    # 检查是否是量化模型
    if input_details['dtype'] == np.uint8:
        input_scale, input_zero_point = input_details['quantization']
        test_images = test_images / input_scale + input_zero_point

    for test_image in test_images:
        input_data = np.expand_dims(test_image, axis=0).astype(input_details['dtype'])
        test_label = test_labels[count]
        interpreter.set_tensor(input_details['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details['index'])
        print(output_data)
        # 计算精度
        accuracy.update_state(test_label, tf.argmax(output_data, axis=1))
        count += 1

    end = time.time()
    print("test tflite on: {} examples, accuracy: {}, time: {}".format(
        count, accuracy.result().numpy(), end-start
    ))

    # 清零
    count = 0
    accuracy.reset_states()




if __name__ == '__main__':

    test_img_paths, test_img_labels = generate_voc_image_label_list(cls="person", \
        voc_label_path=voc_label_path, voc_image_path=voc_image_path, mode="val")
    
    test_images, test_labels = load_data(test_img_paths, test_img_labels)

    # lite_convert('model/voc.h5', quantization="none", save_path="model/voc.tflite", \
    #     image_list=test_images, count=100)

    evaluate_tflite(model_path="model/voc_q.tflite", test_images=test_images, test_labels=test_labels)

    

    

    
    

    
    