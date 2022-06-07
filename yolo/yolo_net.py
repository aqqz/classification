import tensorflow as tf
from keras.layers import Conv2D, Dense, Reshape, GlobalAveragePooling2D, Flatten, Dropout


from data_gen import *

base_model = tf.keras.applications.MobileNet(
    alpha = 0.5,
    input_shape=(224, 224, 3),
    weights="imagenet",
    include_top=False
)

base_model.trainable=False

def yolo_net(input):

    gray2rgb = Conv2D(3, kernel_size=1, strides=1, activation=None)(input)
    x = base_model(gray2rgb, training=False)

    x = GlobalAveragePooling2D()(x)
    x = Dense(S*S*(5+C))(x)
    x = Reshape(target_shape=(S, S, 5+C))(x)

    return x


