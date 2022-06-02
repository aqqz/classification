import tensorflow as tf
import datetime
import os
from utils import *
from loss import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train(train_ds, val_ds, EPOCHS, BATCH_SIZE=32, lr=0.01, save_path='model/model.h5'):
    
    train_ds = train_ds.shuffle(train_ds.cardinality().numpy()).batch(BATCH_SIZE)
    val_ds = val_ds.batch(BATCH_SIZE)

    num_classes = train_ds.element_spec[1].shape[1]

    # 使用预训练权重迁移学习
    base_model = tf.keras.applications.MobileNet(
        input_shape=(224, 224, 3),
        alpha=0.5,
        weights="imagenet",
        include_top=False
    )
    base_model.trainable = False
    

    # 构建模型
    input = tf.keras.layers.Input(shape=(224, 224, 1))
    first_layer = tf.keras.layers.SeparableConvolution2D(
        filters=3,
        kernel_size=1,
        strides=1,
        activation=None,
        name="gray2rgb"
    )(input)
    x = base_model(first_layer, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    output = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(input, output)

    # 微调
    # model = tf.keras.models.load_model("model/voc.h5")
    # model.trainable=True

    model.summary()
    
    # 训练配置
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
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
            logits = model(images)
            loss_value = cls_loss(labels, logits)
        gradients = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss_value)
        train_accuracy(tf.argmax(labels,axis=1), tf.argmax(logits, axis=1))
    
    # 验证阶段
    @tf.function
    def val_step(images, labels):
        logits = model(images)
        loss_value = cls_loss(labels, logits)
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

        # 指标清零
        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()

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
        
    
    model.save(save_path)


def train_locator(train_ds, val_ds, EPOCHS, BATCH_SIZE=32, lr=0.01, save_path='model/model.h5'):
    
    train_ds = train_ds.shuffle(train_ds.cardinality().numpy()).batch(BATCH_SIZE)
    val_ds = val_ds.batch(BATCH_SIZE)

    num_classes = 2

    # 使用预训练权重迁移学习
    # base_model = tf.keras.applications.MobileNet(
    #     input_shape=(224, 224, 3),
    #     alpha=0.5,
    #     weights="imagenet",
    #     include_top=False
    # )
    # base_model.trainable = False
    

    # 构建模型
    # input = tf.keras.layers.Input(shape=(224, 224, 1))
    # first_layer = tf.keras.layers.SeparableConvolution2D(
    #     filters=3,
    #     kernel_size=1,
    #     strides=1,
    #     activation=None,
    #     name="gray2rgb"
    # )(input)
    # x = base_model(first_layer, training=False)
    # x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # output1 = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    # output2 = tf.keras.layers.Dense(4, activation="relu")(x)
    # model = tf.keras.Model(input, [output1, output2])

    # 微调
    model = tf.keras.models.load_model("model/voc.h5")
    model.trainable=True

    model.summary()
    
    # 训练配置
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
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
            pred1, pred2 = model(images)
            pred = tf.concat([pred1, pred2], axis=1)
            loss_value = loc_loss(labels, pred)
        gradients = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss_value)
        train_accuracy(tf.argmax(labels[:, 0:2],axis=1), tf.argmax(pred[:, 0:2], axis=1))
    
    # 验证阶段
    @tf.function
    def val_step(images, labels):
        pred1, pred2 = model(images)
        pred = tf.concat([pred1, pred2], axis=1)
        loss_value = loc_loss(labels, pred)
        val_loss(loss_value)
        val_accuracy(tf.argmax(labels[:, 0:2],axis=1), tf.argmax(pred[:, 0:2], axis=1))
        
    
    # 训练循环
    for epoch in range(EPOCHS):
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
            
        with val_summary_writer.as_default():
            tf.summary.scalar('loss', val_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', val_accuracy.result(), step=epoch)

        # 指标清零
        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()

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
        
    
    model.save(save_path)


if __name__ == '__main__':

    data_root = '/home/taozhi/datasets/flowers' # 训练数据根目录
    print(data_root)
    class_names = os.listdir(data_root)
    print(class_names)

    image_paths, image_labels = genearte_image_list(data_root, class_names)

    train_ds, val_ds = generate_split_dataset(image_paths, image_labels, class_names, split_rate=0.8)

    train(train_ds, val_ds, EPOCHS=50, BATCH_SIZE=32, lr=1e-4, save_path='model/model.h5')

    