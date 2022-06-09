import sys
sys.path.append(".")

from voc.voc_datagen import *
import numpy as np


S = 7
B = 2
C = 20
img_size = 224
grid_size = img_size // S
threshold = 0.5

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
        normal_W = ob_W / img_size
        normal_H = ob_H / img_size
        # print(grid_X, grid_Y, normal_X, normal_Y, normal_W, normal_H)

        # fill label
        ob_vector = image_label[grid_X, grid_Y]
        ob_vector[0]=1
        ob_vector[1:5]=normal_X, normal_Y, normal_W, normal_H
        ob_vector[5+ob_id]=1
        # print(ob_vector)

    return image_label


def post_progress(img_w, img_h, output):
    ob_infos = []
    for i in range(S):
        for j in range(S):
            grid_vector = output[0, i, j]
            tf.print(grid_vector)
            # [ob_exist, x, y, w, h, ..., C]
            if grid_vector[0] > threshold:
                scale_x = img_w / img_size
                scale_y = img_h / img_size
                normal_x = grid_vector[1]
                normal_y = grid_vector[2]
                normal_w = grid_vector[3]
                normal_h = grid_vector[4]
                # tf.print(normal_x, normal_y, normal_w, normal_h)
                # exist object
                ob_x = (normal_x + i) * grid_size
                ob_y = (normal_y + j) * grid_size
                ob_w = normal_w * img_size
                ob_h = normal_h * img_size
                # tf.print(ob_x, ob_y, ob_w, ob_h)
                ob_x *= scale_x
                ob_y *= scale_y
                ob_w *= scale_x
                ob_h *= scale_y
                # tf.print(ob_x, ob_y, ob_w, ob_h)
                xmin = int(ob_x - ob_w / 2)
                ymin = int(ob_y - ob_h / 2) 
                xmax = int(ob_x + ob_w / 2) 
                ymax = int(ob_y + ob_h / 2)
                # tf.print(xmin, ymin, xmax, ymax)
                cls_vector = grid_vector[5:]
                cls = tf.argmax(cls_vector).numpy()
                # tf.print(cls)
                # todo nms
                ob_infos.append([cls, xmin, ymin, xmax, ymax])
    # print(ob_infos)
    return ob_infos


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
    xml_path = os.path.join(voc_annotation_path, '2007_000032.xml')
    test_img = os.path.join(voc_image_path, '2007_000032.jpg')
    img_name, img_w, img_h, ob_infos = parse_xml(xml_path)
    img_label = generate_yolo_label(img_w, img_h, ob_infos)
    # print(img_label)
    output = tf.expand_dims(img_label, axis=0)
    ob_infos = post_progress(img_w, img_h, output)
    draw_box(test_img, ob_infos)

