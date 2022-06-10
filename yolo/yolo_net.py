import tensorflow as tf
from keras.layers import Input, Conv2D, Concatenate
from keras import Model


from data_gen import *

base_model = tf.keras.applications.MobileNet(
    alpha = 1.0,
    input_shape=(224, 224, 3),
    weights="imagenet",
    include_top=False
)

base_model.trainable=False

def yolo_net(input, bbox_num=1, class_num=20):
    # backbone
    gray2rgb = Conv2D(3, kernel_size=1, strides=1, activation=None, name="gray2rgb")(input)
    x = base_model(gray2rgb, training=False)

    # yolo_head
    conf = Conv2D(1*bbox_num, 1, padding="same", activation="sigmoid")(x)
    bbox = Conv2D(4*bbox_num, 1, padding="same", activation="sigmoid")(x)
    prob = Conv2D(class_num, 1, padding="same", activation="softmax")(x)

    output = Concatenate(axis=-1)([conf, bbox, prob])
    return output


if __name__ == '__main__':

    input = Input(shape=(224, 224, 1))
    output = yolo_net(input)
    model = Model(input, output)
    model.summary()

