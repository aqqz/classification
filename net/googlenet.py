import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, \
    BatchNormalization, ReLU, Concatenate, Dropout, AveragePooling2D

def BasicConv2d(input, filters, kernel_size):
    x = Conv2D(filters=filters, kernel_size=kernel_size, padding="same")(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def InceptionV1(input, out_channels1, out_channels2reduce, out_channels2, 
out_channels3reduce, out_channels3, out_channels4):
    branch1_conv = BasicConv2d(input, out_channels1, (1, 1)) # 1*1卷积不会改变输出特征向量的维度
    branch2_conv1 = BasicConv2d(input, out_channels2reduce, (1, 1))
    branch2_conv2 = BasicConv2d(branch2_conv1, out_channels2, (3, 3))
    branch3_conv1 = BasicConv2d(input, out_channels3reduce, (1, 1))
    branch3_conv2 = BasicConv2d(branch3_conv1, out_channels3, (5, 5))
    branch4_pool = MaxPooling2D((3, 3), strides=1, padding="same")(input)
    branch4_conv = BasicConv2d(branch4_pool, out_channels4, (1, 1))

    return Concatenate(axis=-1)([branch1_conv, branch2_conv2, branch3_conv2, branch4_conv])


def googlenet(input, num_classes):
    # input_shape (224, 224, 3)
    x = Conv2D(64, (7, 7), strides=2, padding="same")(input) #padding=same 保证输出特征图维度不变，仅受步距影响
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D((3, 3), strides=2, padding="same")(x)
    
    x = Conv2D(64, (1, 1))(x)
    x = Conv2D(192, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D((3, 3), strides=2, padding="same")(x)

    x = InceptionV1(x, out_channels1=64, out_channels2reduce=96, out_channels2=128,
    out_channels3reduce=16, out_channels3=32, out_channels4=32)
    x = InceptionV1(x, out_channels1=128, out_channels2reduce=128, out_channels2=192, 
    out_channels3reduce=32, out_channels3=96, out_channels4=64)
    x = MaxPooling2D((3, 3), strides=2, padding="same")(x)

    x = InceptionV1(x, out_channels1=192, out_channels2reduce=96, out_channels2=208, 
    out_channels3reduce=16, out_channels3=48, out_channels4=64)
    x = InceptionV1(x, out_channels1=160, out_channels2reduce=112, out_channels2=224, 
    out_channels3reduce=24, out_channels3=64, out_channels4=64)
    x = InceptionV1(x, out_channels1=128, out_channels2reduce=128, out_channels2=256, 
    out_channels3reduce=24, out_channels3=64, out_channels4=64)
    x = InceptionV1(x, out_channels1=112, out_channels2reduce=144, out_channels2=288, 
    out_channels3reduce=32, out_channels3=64, out_channels4=64)
    x = InceptionV1(x, out_channels1=256, out_channels2reduce=160, out_channels2=320, 
    out_channels3reduce=32, out_channels3=128, out_channels4=128)
    x = MaxPooling2D((3, 3), strides=2, padding="same")(x)

    x = InceptionV1(x, out_channels1=256, out_channels2reduce=160, out_channels2=320, 
    out_channels3reduce=32, out_channels3=128, out_channels4=128)
    x = InceptionV1(x, out_channels1=384, out_channels2reduce=192, out_channels2=384, 
    out_channels3reduce=48, out_channels3=128, out_channels4=128)
    x = AveragePooling2D((7, 7), strides=1)(x)

    x = Dropout(rate=0.4)(x)
    x = Flatten()(x)
    return Dense(num_classes, activation="softmax")(x)
    

