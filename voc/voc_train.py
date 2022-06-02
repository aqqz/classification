from voc_datagen import *
import sys
sys.path.append('.')

import train

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':

    test_class = 'person' # voc数据集只能测试二分类
    class_names = [test_class, 'no'+ test_class]
    print(class_names)

    train_img_paths, train_img_labels = generate_paths_labels_list(
        test_class, voc_label_path, voc_image_path, mode="train", task="loc")

    val_img_paths, val_img_labels = generate_paths_labels_list(
        test_class, voc_label_path, voc_image_path, mode="val", task="loc")

    train_ds = generate_dataset(train_img_paths, train_img_labels)
    val_ds = generate_dataset(val_img_paths, val_img_labels)

    print("training on {} examples, validating on {} examples\n".format( \
        train_ds.cardinality().numpy(),val_ds.cardinality().numpy()))


    train.train(train_ds, val_ds, EPOCHS=20, BATCH_SIZE=32, lr=1e-5, save_path="model/voc.h5")