import tensorflow as tf
from keras.layers import Conv2D, Dense, Flatten, GlobalAveragePooling2D

base_model = tf.keras.applications.MobileNet(
    alpha=0.5,
    input_shape=(224, 224, 3),
    weights='imagenet',
    include_top=False,
)
base_model.trainable = False

def net(input, num_classes):
    gray2rgb = Conv2D(3, kernel_size=1, activation=None)(input)
    
    x = base_model(gray2rgb, training=False)
    x = GlobalAveragePooling2D()(x)

    out = Dense(num_classes, activation="softmax")(x)

    return out
    

