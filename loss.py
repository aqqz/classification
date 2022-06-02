import tensorflow as tf

# loss function for cls
def cls_loss(y_true, y_pred):
    # tf.print("y_true: ", y_true)
    # tf.print("y_pred: ", y_pred)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    loss_value = loss_fn(y_true, y_pred)
    return loss_value

# loss function for cls+loc
def loc_loss(y_true, y_pred):
    # tf.print("y_true: ", y_true)
    # tf.print("y_pred: ", y_pred)
    cls_loss_fn = tf.keras.losses.CategoricalCrossentropy()
    cls_loss = cls_loss_fn(y_true[:, 0:2], y_pred[:, 0:2])
    loc_loss_fn = tf.keras.losses.MeanSquaredError()
    loc_loss = loc_loss_fn(y_true[:, 2:], y_pred[:, 2:])
    return cls_loss + loc_loss