from ast import parse
import os
import random
import tensorflow as tf

import sys
sys.path.append('.')
import utils
import xml.etree.ElementTree as et

voc_label_path = '/home/taozhi/datasets/VOCdevkit/VOC2012/ImageSets/Main'
voc_image_path = '/home/taozhi/datasets/VOCdevkit/VOC2012/JPEGImages'
voc_annotation_path = '/home/taozhi/datasets/VOCdevkit/VOC2012/Annotations'

voc_class_list = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]


def parse_xml(xml_path):
    # 解析annotation，返回目标信息
    tree = et.parse(xml_path)
    root = tree.getroot()
    
    img_name = root.find('filename').text 

    size = root.find('size')
    img_w = eval(size.find('width').text)
    img_h = eval(size.find('height').text)

    obs = root.findall('object')
    ob_infos = []
    for ob in obs:
        ob_id = voc_class_list.index(ob.find('name').text)
        bbox = ob.find('bndbox')
        xmin = eval(bbox.find('xmin').text)
        ymin = eval(bbox.find('ymin').text)
        xmax = eval(bbox.find('xmax').text)
        ymax = eval(bbox.find('ymax').text)
        ob_infos.append([ob_id, xmin, ymin, xmax, ymax])

    return img_name, img_w, img_h, ob_infos


def generate_paths_labels_list(cls, voc_label_path, voc_image_path, mode="train"):    
    img_paths = []
    img_labels = []
    
    label_path = os.path.join(voc_label_path, cls + '_' + mode + '.txt')
    print("reading..." + label_path + "\n")
    
    # read txt
    f = open(label_path, "r")
    
    lines = f.readlines()
    for line in lines:
        imgname = line[0:11] # fix here
        imglabel = eval(line[-3:-1])
        img_path = os.path.join(voc_image_path, imgname +'.jpg')
        # generate label
        if imglabel == 1:
            img_label = [1, 0]
        elif imglabel == -1:
            img_label = [0, 1]
        else:
            continue
        # fill imgpath, imglabel list
        img_paths.append(img_path)
        img_labels.append(img_label)


    # # shuffle 
    # random.seed(123)
    # random.shuffle(img_paths)
    # random.seed(123)
    # random.shuffle(img_labels)
    
    return img_paths, img_labels


def generate_dataset(image_paths, image_labels):

    image_dataset = tf.data.Dataset.from_tensor_slices(image_paths).map(utils.load_image)
    label_dataset = tf.data.Dataset.from_tensor_slices(image_labels)
    dataset = tf.data.Dataset.zip((image_dataset, label_dataset))

    return dataset



if __name__ == '__main__':
    
    xml_path = os.path.join(voc_annotation_path, '2007_000032.xml')
    img_name, img_w, img_h, ob_infos = parse_xml(xml_path)
    img_path = os.path.join(voc_image_path, img_name)
    utils.draw_box(img_path, ob_infos)