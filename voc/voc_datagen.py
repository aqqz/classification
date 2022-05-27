import os
import random
import tensorflow as tf

import sys
sys.path.append('.')
import utils

def generate_voc_image_label_list(cls, voc_label_path, voc_image_path, mode="train"):    
    img_paths = []
    img_labels = []
    
    label_path = os.path.join(voc_label_path, cls + '_' + mode + '.txt')
    print("reading..." + label_path)
    
    # read txt
    f = open(label_path, "r")
    
    lines = f.readlines()
    for line in lines:
        imgname = line[0:6]
        imglabel = eval(line[-3:-1])
        # fill imgpath, imglabel list
        img_path = os.path.join(voc_image_path, imgname +'.jpg')
        if imglabel == 1:
            img_label = 0
        else:
            img_label = 1
        img_paths.append(img_path)
        img_labels.append(img_label)


    # shuffle
    random.seed(123)
    random.shuffle(img_paths)
    random.seed(123)
    random.shuffle(img_labels)
    
    return img_paths, img_labels


def generate_dataset(image_paths, image_labels):

    image_dataset = tf.data.Dataset.from_tensor_slices(image_paths).map(utils.load_image)
    label_dataset = tf.data.Dataset.from_tensor_slices(image_labels).map(lambda x: tf.one_hot(x, 2))
    dataset = tf.data.Dataset.zip((image_dataset, label_dataset))

    return dataset




    

    