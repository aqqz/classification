import os
import random
import tensorflow as tf

import sys
sys.path.append('.')
import utils
import cv2

voc_label_path = '/home/taozhi/datasets/VOCdevkit/VOC2012/ImageSets/Main'
voc_image_path = '/home/taozhi/datasets/VOCdevkit/VOC2012/JPEGImages'
voc_annotation_path = '/home/taozhi/datasets/VOCdevkit/VOC2012/Annotations'

voc_class_list = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

def generate_paths_labels_list(cls, voc_label_path, voc_image_path, mode="train", task="cls"):    
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
        if task=="cls":
            if imglabel == 1:
                img_label = [1, 0]
            elif imglabel == -1:
                img_label = [0, 1]
            else:
                continue
        elif task=="loc":
            img_annotation_path = os.path.join(voc_annotation_path, imgname + '.xml')
            if imglabel==1:
                img_w, img_h, xmin, ymin, xmax, ymax = \
                    utils.parse_xml_cls_loc(img_annotation_path, class_name=cls)
                img_label=[1, 0, xmin/img_w, ymin/img_h, xmax/img_w, ymax/img_h]
                # test
                # img = cv2.imread(img_path)
                # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(255, 0, 0), thickness=1, lineType=1)
                # cv2.imwrite(imgname+'.jpg', img)
            elif imglabel==-1:
                img_label=[0, 1, 0, 0, 0, 0]
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




    

    