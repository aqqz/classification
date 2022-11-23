import tensorflow as tf
import datetime
import os
from utils import *
from loss import *
from model import net
from net.lenet5 import lenet5

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train(train_ds, val_ds, EPOCHS, BATCH_SIZE=32, optim="sgd", lr=0.01):
    """
    TensorFlow自定义训练
    """
    # 准备数据
    train_ds = train_ds.shuffle(train_ds.cardinality().numpy()).batch(BATCH_SIZE)
    val_ds = val_ds.batch(BATCH_SIZE)

    num_classes = train_ds.element_spec[1].shape[1]

    # 构建模型
    input = tf.keras.layers.Input(shape=(224, 224, 1))
    output = lenet5(input, num_classes)
    model = tf.keras.Model(input, output)
    model.summary()
    
    # 配置优化器，学习率
    if optim=="sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    if optim=="adam":
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
            logits = model(images, training=True)
            loss_value = cls_loss(labels, logits)
        gradients = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss_value)
        train_accuracy(tf.argmax(labels,axis=1), tf.argmax(logits, axis=1))
    
    # 验证阶段
    @tf.function
    def val_step(images, labels):
        logits = model(images, training=False)
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
        
    # 保存模型
    model.save("model/model.h5")




if __name__ == '__main__':

    data_root = '/home/taozhi/datasets/ds' # 训练数据根目录
    print(data_root)
    class_names = os.listdir(data_root)
    print(class_names)

    image_paths, image_labels = genearte_image_list(data_root, class_names)

    train_ds, val_ds = generate_split_dataset(image_paths, image_labels, class_names, split_rate=0.7)

    train(train_ds, val_ds, EPOCHS=50, BATCH_SIZE=32, optim="sgd", lr=0.01)    