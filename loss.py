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
    cls_loss = -(y_true[:, 0] * tf.math.log(y_pred[:, 0]) + y_true[:, 1] * tf.math.log(y_pred[:, 1]))
    cls_loss = tf.reduce_mean(cls_loss)
    flags = tf.argmax(1-y_true[:, 0:2], axis=1)
    flags = tf.expand_dims(flags, axis=-1)
    flags = tf.tile(flags, [1, 4])
    flags = tf.cast(flags, tf.float32)
    loc_loss = tf.square(flags*(y_true[:, 2:] - y_pred[:, 2:])) # (?, 4)
    loc_loss = tf.reduce_mean(loc_loss)
    return cls_loss + 5 * loc_loss