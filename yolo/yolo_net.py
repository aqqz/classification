import tensorflow as tf
from keras.layers import Conv2D, Dense, Reshape, GlobalAveragePooling2D, Flatten, Dropout, LeakyReLU, MaxPooling2D


from data_gen import *

base_model = tf.keras.applications.MobileNet(
    alpha = 0.5,
    input_shape=(224, 224, 3),
    weights="imagenet",
    include_top=False
)

base_model.trainable=False

# def yolo_net(input):

#     gray2rgb = Conv2D(3, kernel_size=1, strides=1, activation=None)(input)
#     x = base_model(gray2rgb, training=False)

#     x = GlobalAveragePooling2D()(x)
#     x = Dense(S*S*(5+C), activation="sigmoid")(x)
#     x = Reshape(target_shape=(S, S, 5+C))(x)

#     return x

def conv(input, filters, kernel_size, strides=1, padding="same"):
    x = Conv2D(filters, kernel_size, strides, padding)(input)
    x = LeakyReLU(alpha=0.1)(x)
    return x


def yolo_net(input):

    gray2rgb = Conv2D(3, kernel_size=1, strides=1, activation=None)(input)

    x = conv(gray2rgb, 64, 7, 2)
    x = MaxPooling2D(2, 2)(x)

    x = conv(x, 192, 3)
    x = MaxPooling2D(2, 2)(x)

    x = conv(x, 128, 1)
    x = conv(x, 256, 3)
    x = conv(x, 256, 1)
    x = conv(x, 512, 3)
    x = MaxPooling2D(2, 2)(x)

    for i in range(4):
        x = conv(x, 256, 1)
        x = conv(x, 512, 3)
    x = conv(x, 512, 1)
    x = conv(x, 1024, 3)
    x = MaxPooling2D(2, 2)(x)

    for i in range(2):
        x = conv(x, 512, 1)
        x = conv(x, 1024, 3)
    x = conv(x, 1024, 3)
    x = conv(x, 1024, 3, 2)

    x = conv(x, 1024, 3)
    x = conv(x, 1024, 3)

    x = conv(x, 256, 3)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(S*S*(5+C))(x)
    x = Reshape(target_shape=(S, S, 5+C))(x)
    return x



