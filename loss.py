import tensorflow as tf

def loss(y_true, y_pred):
    # tf.print("y_true: ", y_true)
    # tf.print("y_pred: ", y_pred)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    loss_value = loss_fn(y_true, y_pred)
    return loss_value