import tensorflow as tf
import datetime
import os
import random
from net.alexnet import AlexNet
from net.lenet5 import lenet5
from net.vggnet import vggnet
from net.googlenet import googlenet
from net.resnet import resnet

def load_image(image_path):
    raw = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(raw, channels=1)
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32)
    img /= 255.0
    return img


def to_one_hot(image_label):
    return tf.one_hot(image_label, len(class_names))


def genearte_image_list(data_root):
    class_dirs = [os.path.join(data_root, class_name) for class_name in class_names]
    
    image_paths = []
    image_labels = []
    for class_dir in class_dirs:
        files = os.listdir(class_dir)
        for file in files:
            # 过滤不是.jpg后缀文件
            if file.endswith('.jpg'):
                image_path = os.path.join(class_dir, file)
                image_label = class_names.index(class_dir[len(data_root)+1:])
                image_paths.append(image_path)
                image_labels.append(image_label)
    
    # 此处打乱提高低负载设备性能
    random.seed(123)
    random.shuffle(image_paths)
    random.seed(123)
    random.shuffle(image_labels)
        
    return image_paths, image_labels


def generate_split_dataset(image_paths, image_labels, split_rate=0.8):
    image_dataset = tf.data.Dataset.from_tensor_slices(image_paths).map(load_image)
    label_dataset = tf.data.Dataset.from_tensor_slices(image_labels).map(to_one_hot)
    dataset = tf.data.Dataset.zip((image_dataset, label_dataset))
    
    # 划分数据集
    data_len = dataset.cardinality().numpy()
    train_len = int(data_len*split_rate)
    val_len = data_len - train_len
    
    train_dataset = dataset.take(train_len)
    val_dataset = dataset.skip(train_len).take(val_len)

    print(f"\nloaded {data_len} images.")
    print(f"training on {train_len} images, validating on {val_len} images.\n")

    return train_dataset, val_dataset


def train(train_ds, val_ds, EPOCHS, BATCH_SIZE=32):
    
    train_ds = train_ds.shuffle(train_ds.cardinality().numpy()).batch(BATCH_SIZE)
    val_ds = val_ds.batch(BATCH_SIZE)

    # 构建模型
    # input = tf.keras.layers.Input(shape=(224, 224, 3))
    # output = resnet(input, num_classes=len(class_names))
    # model = tf.keras.Model(input, output)
    model = lenet5(len(class_names))

    model.summary()
    # 训练配置
    loss = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    
    # 记录指标
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.Accuracy(name="train_acc")
    val_loss = tf.keras.metrics.Mean(name="val_loss")
    val_accuracy = tf.keras.metrics.Accuracy(name="val_acc")

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'runs/' + current_time + '/train'
    val_log_dir = 'runs/' + current_time + '/val'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)
    
    # 训练阶段
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            logits = model(images, training=True)
            loss_value = loss(labels, logits)
        gradients = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss_value)
        # tf.print("labels: ", labels)
        # tf.print("logits: ", logits)
        train_accuracy(tf.argmax(labels,axis=1), tf.argmax(logits, axis=1))
    
    # 验证阶段
    @tf.function
    def val_step(images, labels):
        logits = model(images, training=False)
        loss_value = loss(labels, logits)
        val_loss(loss_value)
        val_accuracy(tf.argmax(labels,axis=1), tf.argmax(logits, axis=1))
        
    
    # 训练循环
    for epoch in range(EPOCHS):
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
            
        with val_summary_writer.as_default():
            tf.summary.scalar('loss', val_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', val_accuracy.result(), step=epoch)

        for step, (images, labels) in enumerate(train_ds):
            train_step(images, labels)

        for step, (images, labels) in enumerate(val_ds):
            val_step(images, labels)
        
        
        pattern = '{:.3f}'
        print(
            'Epoch ' + '{}'.format(epoch+1),
            'Loss: ' + pattern.format(train_loss.result()),
            'Accuracy: ' + pattern.format(train_accuracy.result()),
            'Val Loss: ' + pattern.format(val_loss.result()), 
            'Val Accuracy: ' + pattern.format(val_accuracy.result())
        )
        
        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()


    # model.compile(
    #     loss="categorical_crossentropy",
    #     optimizer="adam",
    #     metrics=['accuracy']
    # )
    # model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
    
    model.save("model/model.h5")


if __name__ == '__main__':

    data_root = '/home/taozhi/datasets/dogs_vs_cats/train' # 训练数据根目录
    print(data_root)
    class_names = os.listdir(data_root)
    print(class_names)

    image_paths, image_labels = genearte_image_list(data_root)

    train_ds, val_ds = generate_split_dataset(image_paths, image_labels, split_rate=0.8)

    train(train_ds, val_ds, EPOCHS=50, BATCH_SIZE=32)

    