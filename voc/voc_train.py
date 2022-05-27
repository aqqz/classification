from voc_datagen import generate_dataset, generate_voc_image_label_list
import sys
sys.path.append('.')

import train

voc_label_path = '/home/taozhi/datasets/VOCdevkit/VOC2007/ImageSets/Main'
voc_image_path = '/home/taozhi/datasets/VOCdevkit/VOC2007/JPEGImages'
voc_annotation_path = '/home/taozhi/datasets/VOCdevkit/VOC2007/Annotations'

voc_class_list = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

if __name__ == '__main__':

    test_class = 'person' # voc数据集只能测试二分类
    class_names = [test_class, 'no'+ test_class]
    print(class_names)

    train_img_paths, train_img_labels = \
        generate_voc_image_label_list(test_class, voc_label_path, voc_image_path, mode="train")

    val_img_paths, val_img_labels = \
        generate_voc_image_label_list(test_class, voc_label_path, voc_image_path, mode="val")

    train_ds = generate_dataset(train_img_paths, train_img_labels)
    val_ds = generate_dataset(val_img_paths, val_img_labels)

    print("training on {} examples, validating on {} examples".format( \
        train_ds.cardinality().numpy(),val_ds.cardinality().numpy()))


    train.train(train_ds, val_ds, EPOCHS=50, BATCH_SIZE=32, lr=1e-4, save_path="model/voc.h5")