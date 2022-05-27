import tensorflow as tf
from voc.voc_datagen import *
from utils import load_data



def lite_convert(model_path, quantization="none", save_path="model/model.tflite", image_list=[], count=100):

    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantization=='none':
        # save sample for quantize aware training
        pass
    # https://www.tensorflow.org/lite/performance/post_training_integer_quant#%E4%BD%BF%E7%94%A8%E6%B5%AE%E7%82%B9%E5%9B%9E%E9%80%80%E9%87%8F%E5%8C%96%E8%BF%9B%E8%A1%8C%E8%BD%AC%E6%8D%A2
    elif quantization=='int8':
        def representative_dataset():
            for data in tf.data.Dataset.from_tensor_slices(image_list).batch(1).take(count):
                yield [tf.dtypes.cast(data, tf.float32)]
        
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()
        
    with open(save_path, "wb") as f:
        f.write(tflite_model)






if __name__ == '__main__':

    test_img_paths, test_img_labels = generate_voc_image_label_list(cls="person", \
        voc_label_path=voc_label_path, voc_image_path=voc_image_path, mode="val")
    
    test_images, test_labels = load_data(test_img_paths, test_img_labels)

    lite_convert('model/voc.h5', quantization="none", save_path="model/voc.tflite", \
        image_list=test_images, count=100)

    

    
    

    
    