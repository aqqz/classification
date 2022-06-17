import tensorflow as tf

def cls_loss(y_true, y_pred):
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    loss_value = loss_fn(y_true, y_pred)
    return loss_value