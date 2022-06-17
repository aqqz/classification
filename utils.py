import tensorflow as tf
import os
import random
import numpy as np
import cv2
    
        

def load_image(image_path):
    """
    TensorFlow读取数据，返回image_tensor [0, 1]
    """
    raw = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(raw, channels=1)
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32)    
    img = img / 255
    return img


def to_one_hot(image_label, class_names):
    """
    标签转one-hot编码
    0 -> [1, 0]
    1 -> [0, 1]
    """
    return tf.one_hot(image_label, len(class_names))


def genearte_image_list(data_root, class_names):
    """
    从分类好的数据根目录生成图片路径列表，图片标签列表
    """
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


def load_data(datasets):
    """
    输入TensorFlow Dataset, 返回Numpy ndarray类型图像和标签
    """
    images = []
    labels = []
    print("loading data from datasets...\n")
    
    for step, (image, label) in enumerate(datasets):
        images.append(image)
        labels.append(label)
    images = np.array(images)
    labels = np.array(labels)

    return images, labels



def generate_split_dataset(image_paths, image_labels, class_names, split_rate=0.8):
    """
    输入图片路径列表和图片标签列表，根据训练验证比例，划分数据集，返回训练集、验证集
    """
    image_dataset = tf.data.Dataset.from_tensor_slices(image_paths).map(load_image)
    label_dataset = tf.data.Dataset.from_tensor_slices(image_labels).map(
        lambda x: to_one_hot(x, class_names))
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



def draw_box(img_path, ob_infos):
    """
    输入图片路径和坐标框信息，绘制bounding box
    """
    img = cv2.imread(img_path)
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for ob_info in ob_infos:
        xmin, ymin, xmax, ymax = ob_info[1], ob_info[2], ob_info[3], ob_info[4]
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(0, 0, 255), thickness=2, lineType=1)
        cv2.putText(img, text=str(ob_info[0]), org=(xmin-3, ymin-3), fontFace=1, fontScale=1, color=(0, 0, 255), thickness=1, lineType=1)
    cv2.imwrite('test.jpg', img)
