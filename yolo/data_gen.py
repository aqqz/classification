import sys
sys.path.append(".")

from voc.voc_datagen import *
import numpy as np


S = 4
C = 20
img_size = 224
grid_size = img_size // S

def generate_yolo_label(img_w, img_h, ob_infos, cell=S):
    image_label = np.zeros((cell, cell, 5+C), dtype="float32")
    
    for ob_info in ob_infos:
        # [ob_id, xmin, ymin, xmax, ymax]
        ob_id, xmin, ymin, xmax, ymax = ob_info[0], ob_info[1], ob_info[2], ob_info[3], ob_info[4]
        ob_X = (xmin + xmax) / 2
        ob_Y = (ymin + ymax) / 2
        ob_W = xmax - xmin
        ob_H = ymax - ymin
        scale_X = img_w / img_size
        scale_Y = img_h / img_size
        # print(scale_X, scale_Y)
        ob_X /= scale_X
        ob_Y /= scale_Y
        ob_W /= scale_X
        ob_H /= scale_Y
        # print(ob_X, ob_Y, ob_W, ob_H, img_w, img_h)
        # find grid_X, grid_Y    
        grid_X = int(ob_X // grid_size)
        grid_Y = int(ob_Y // grid_size)
        normal_X = ob_X / grid_size - grid_X
        normal_Y = ob_Y / grid_size - grid_Y
        normal_W = ob_W / grid_size
        normal_H = ob_H / grid_size
        # print(grid_X, grid_Y, normal_X, normal_Y, normal_W, normal_H)

        # fill label
        ob_vector = image_label[grid_X, grid_Y]
        ob_vector[0]=1
        ob_vector[1:5]=normal_X, normal_Y, normal_W, normal_H
        ob_vector[5+ob_id]=1
        # print(ob_vector)

    return image_label


def generate_image_list(txt_file):
    image_paths = []
    image_labels = []
    
    f = open(txt_file)
    lines = f.readlines()
    for line in lines:
        line = line[:-1]
        
        xml_path = os.path.join(voc_annotation_path, line+'.xml')
        # parse_xml
        img_name, img_w, img_h, ob_infos = parse_xml(xml_path)
        img_path = os.path.join(voc_image_path, img_name)
        
        # generate_yolo_label
        img_label = generate_yolo_label(img_w, img_h, ob_infos)
        
        # fill image_paths image_labels
        image_paths.append(img_path)
        image_labels.append(img_label)


    # shuffle
    if str(txt_file).endswith('train.txt'):
        random.seed(123)
        random.shuffle(image_paths)
        random.seed(123)
        random.shuffle(image_labels)


    return image_paths, image_labels




if __name__ == '__main__':
    train_image_paths, train_image_labels = generate_image_list(os.path.join(
        voc_label_path, 'train.txt'))
    val_image_paths, val_image_labels = generate_image_list(os.path.join(
        voc_label_path, 'val.txt'))

    train_ds = generate_dataset(train_image_paths, train_image_labels)
    val_ds = generate_dataset(val_image_paths, val_image_labels)

    tf.print(train_ds)
    tf.print(val_ds)