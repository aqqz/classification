import tensorflow as tf
import os
import random
import numpy as np

def load_image(image_path):
    raw = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(raw, channels=1)
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32)
    img /= 255
    return img


def to_one_hot(image_label, class_names):
    return tf.one_hot(image_label, len(class_names))


def genearte_image_list(data_root, class_names):
    class_dirs = [os.path.join(data_root, class_name) for class_name in class_names]
    
    image_paths = []
    image_labels = []
    for class_dir in class_dirs:
        files = os.listdir(class_dir)
        for file in files:
            # 过滤不是.jpg后缀文件
            if file.endswith('.jpg'):
                image_path = os.path.join(class_dir, file)
                image_label = class_names.index(class_dir[len(data_root)+1:])
                image_paths.append(image_path)
                image_labels.append(image_label)
    
    # 此处打乱提高低负载设备性能
    random.seed(123)
    random.shuffle(image_paths)
    random.seed(123)
    random.shuffle(image_labels)
        
    return image_paths, image_labels


def load_data(image_paths, image_labels):
    test_images = []
    test_labels = []

    for path in image_paths:
        test_images.append(load_image(path))

    for label in image_labels:
        test_labels.append(label)

    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    return test_images, test_labels



def generate_split_dataset(image_paths, image_labels, class_names, split_rate=0.8):
    image_dataset = tf.data.Dataset.from_tensor_slices(image_paths).map(load_image)
    label_dataset = tf.data.Dataset.from_tensor_slices(image_labels).map(lambda x: to_one_hot(x, class_names))
    dataset = tf.data.Dataset.zip((image_dataset, label_dataset))
    
    # 划分数据集
    data_len = dataset.cardinality().numpy()
    train_len = int(data_len*split_rate)
    val_len = data_len - train_len
    
    train_dataset = dataset.take(train_len)
    val_dataset = dataset.skip(train_len).take(val_len)

    print(f"\nloaded {data_len} images.")
    print(f"training on {train_len} images, validating on {val_len} images.\n")

    return train_dataset, val_dataset

