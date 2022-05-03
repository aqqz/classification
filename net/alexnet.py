import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, ReLU, Softmax

def AlexNet(num_classes, input_shape):
    return tf.keras.Sequential([
        # 卷积，批量归一化，激活
        Conv2D(96, (11, 11), strides=4, input_shape=input_shape, name="conv1"),
        BatchNormalization(name="bn1"), # 增加bn层有利于梯度下降
        ReLU(name="relu1"),
        MaxPooling2D(3, strides=2, name="pool1"),

        Conv2D(256, (5, 5), padding="same", name="conv2"),
        BatchNormalization(name="bn2"),
        ReLU(name="relu2"),
        MaxPooling2D(3, strides=2, name="pool2"),
        
        Conv2D(384, (3, 3), padding="same", name="conv3"),
        BatchNormalization(name="bn3"),
        ReLU(name="relu3"),

        Conv2D(384, (3, 3), padding="same", name="conv4"),
        BatchNormalization(name="bn4"),
        ReLU(name="relu4"),

        Conv2D(256, (3, 3), padding="same", name="conv5"),
        BatchNormalization(name="bn5"),
        ReLU(name="relu5"),
        MaxPooling2D(3, strides=2, name="pool3"),

        Flatten(),
        Dense(4096, name="fc1"),
        ReLU(name="relu6"),
        Dropout(0.5, name="dropout1"),# 随机失活部分神经元
        
        Dense(4096, name="fc2"),
        ReLU(name="relu7"),
        Dropout(0.5, name="dropout2"),
        
        Dense(num_classes, name="fc3"),
        Softmax(name="softmax"),
    ])
