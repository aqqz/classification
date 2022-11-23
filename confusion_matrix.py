import os
import tensorflow as tf
from utils import load_data, genearte_image_list, generate_split_dataset
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    data_root = '/home/taozhi/datasets/flowers' # 训练数据根目录
    class_names = os.listdir(data_root) 
    print(class_names)

    image_paths, image_labels = genearte_image_list(data_root, class_names)
    train_ds, val_ds = generate_split_dataset(image_paths, image_labels, class_names, split_rate=0.8)
    
    test_images, test_labels = load_data(val_ds) #导入验证集数据

    net = tf.keras.models.load_model("model/model.h5")
    predicts = net.predict_on_batch(test_images)

    y_true = [class_names[np.argmax(label)] for label in test_labels] # 标签
    y_pred = [class_names[np.argmax(predict)] for predict in predicts] # 预测   

    C = confusion_matrix(y_true, y_pred, labels=class_names)

    plt.subplot(2, 1, 1)
    plt.matshow(C, cmap=plt.cm.Reds)
    plt.colorbar()

    accuracy = []
    for i in range(len(C)):
        sum = 0
        for j in range(len(C)):
            plt.annotate(C[j, i], xy=(i, j), horizontalalignment="center", verticalalignment="center")
            sum += C[i, j]
        accuracy.append(C[i, i]/sum)
    print(accuracy)
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks(range(0,5), labels=['tulip','daisy','dandelion','sunflower','rose'], rotation="90") # 将x轴或y轴坐标，刻度 替换为文字/字符
    plt.show()
    plt.savefig('confusion_matrix.png')

    plt.subplot(1, 1, 1)
    rect = plt.bar(class_names, accuracy)
    plt.bar_label(rect)
    plt.show()
    plt.savefig('accuracy.png')
    