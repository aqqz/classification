import tensorflow as tf
from yolo_net import yolo_net
from yolo_loss import yolo_loss
from data_gen import *
import datetime


def train_yolo(train_ds, val_ds, EPOCHS, BATCH_SIZE=32, lr=0.01, optim="sgd", save_path="model/yolo.h5"):

    # 载入模型
    input = tf.keras.layers.Input(shape=(224, 224, 1))
    output = yolo_net(input)
    model = tf.keras.models.Model(input, output)

    model.summary()

    # 配置优化器、学习率
    if optim=="sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    elif optim=="adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # 训练数据处理
    train_ds = train_ds.shuffle(train_ds.cardinality().numpy()).batch(BATCH_SIZE)
    val_ds = val_ds.batch(BATCH_SIZE)

    # 记录指标
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    val_loss = tf.keras.metrics.Mean(name="val_loss")
    train_acc = tf.kears.metrics.Accuracy(name="train_acc")
    val_acc = tf.keras.metrics.Accuracy(name="val_acc")

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'runs/' + current_time + '/train'
    val_log_dir = 'runs/' + current_time + '/val'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            logits = model(images)
            loss_value = yolo_loss(labels, logits)
        gradients = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss_value)
        train_acc()

    @tf.function
    def val_step(images, labels):
        logits = model(images)
        loss_value = yolo_loss(labels, logits)
        val_loss(loss_value)
        val_acc()


    # 训练循环
    for epoch in range(EPOCHS):

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_acc.result(), step=epoch)
            
        with val_summary_writer.as_default():
            tf.summary.scalar('loss', val_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', val_acc.result(), step=epoch)

        train_loss.reset_states()
        val_loss.reset_states()
        train_acc.reset_states()
        val_acc.reset_states()


        for step, (images, labels) in enumerate(train_ds):
            train_step(images, labels)

        for step, (images, labels) in enumerate(val_ds):
            val_step(images,labels)

        pattern = '{:.3f}'
        print(
            'Epoch ' + '{}'.format(epoch+1),
            'Loss: ' + pattern.format(train_loss.result()),
            'Accuracy: ' + pattern.format(train_acc.result()),
            'Val Loss: ' + pattern.format(val_loss.result()), 
            'Val Accuracy: ' + pattern.format(val_acc.result())
        )

    model.save(save_path)


if __name__ == '__main__':


    train_image_paths, train_image_labels = generate_image_list(os.path.join(
        voc_label_path, 'train.txt'))
    val_image_paths, val_image_labels = generate_image_list(os.path.join(
        voc_label_path, 'val.txt'))

    train_ds = generate_dataset(train_image_paths, train_image_labels)
    val_ds = generate_dataset(val_image_paths, val_image_labels)

    print("training on {} images, validating on {} images".format(
        train_ds.cardinality().numpy(), val_ds.cardinality().numpy()))


    train_yolo(train_ds, val_ds, EPOCHS=10, BATCH_SIZE=32, lr=0.01, optim="sgd", save_path="model/yolo.h5")