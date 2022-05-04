import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, ReLU, Dropout

def conv(input, filters, kernel_size, name):
    x = Conv2D(filters, kernel_size, padding="same", name=name)(input)
    x = BatchNormalization()(x)
    return ReLU()(x)

def fc(input, units, name):
    x = Dense(units, name=name)(input)
    x = ReLU()(x)
    return Dropout(0.5)(x)

# 使用functional API
def vggnet(input, num_classes):
    # input_shape (224, 224, 3)
    x = conv(input, 64, (3, 3), name="conv1")
    x = conv(x,     64, (3, 3), name="conv2")
    x = MaxPooling2D(2, 2, name="pool1")(x)

    x = conv(x,    128, (3, 3), name="conv3")
    x = conv(x,    128, (3, 3), name="conv4")
    x = MaxPooling2D(2, 2, name="pool2")(x)

    x = conv(x,    256, (3, 3), name="conv5")
    x = conv(x,    256, (3, 3), name="conv6")
    x = conv(x,    256, (3, 3), name="conv7")
    x = MaxPooling2D(2, 2, name="pool3")(x)

    x = conv(x,    512, (3, 3), name="conv8")
    x = conv(x,    512, (3, 3), name="conv9")
    x = conv(x,    512, (3, 3), name="conv10")
    x = MaxPooling2D(2, 2, name="pool4")(x)

    x = conv(x,    512, (3, 3), name="conv11")
    x = conv(x,    512, (3, 3), name="conv12")
    x = conv(x,    512, (3, 3), name="conv13")
    x = MaxPooling2D(2, 2, name="pool5")(x)

    x = Flatten()(x)
    x = fc(x, 4096, name="fc1")
    x = fc(x, 4096, name="fc2")
    return Dense(num_classes, activation="softmax")(x)