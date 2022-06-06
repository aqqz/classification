import tensorflow as tf


def yolo_loss(y_true, y_pred):

    lambda_coord=5
    lambda_noobj=0.5
    epr = 1e-6

    # [batch, S, S, 5+C]
    # [:, :, :, ob, x, y, w, h, ...]
    # tf.print("y_true: ", y_true)
    # tf.print("y_pred: ", y_pred)
    ob_exist = y_true[..., 0] #(?, 4, 4)
    # tf.print(ob_exist)
    
    # 正样本中心点定位损失
    loc_loss = ob_exist * (tf.square(y_true[..., 1] - y_pred[..., 1]) + tf.square(y_true[..., 2] - y_pred[..., 2])) #(?, 4, 4)
    loc_loss = tf.reduce_mean(loc_loss)
    # tf.print(loc_loss)
    
    # 正样本宽高比损失
    scale_loss = ob_exist * (tf.square(tf.sqrt(y_true[..., 3]+epr)-tf.sqrt(y_pred[..., 3]+epr)) \
        + tf.square(tf.sqrt(y_true[..., 4]+epr)-tf.sqrt(y_pred[..., 4]+epr))) # (?, 4, 4)
    scale_loss = tf.reduce_mean(scale_loss)
    # tf.print(scale_loss)
    
    # 正样本网格分类损失
    positive_loss = ob_exist * tf.square(y_true[..., 0] - y_pred[..., 0]) #(?, 4, 4)
    positive_loss = tf.reduce_mean(positive_loss)
    # tf.print(positive_loss)

    # 负样本网格分类损失
    negative_loss = (1-ob_exist) * tf.square(y_true[..., 0] - y_pred[..., 0]) #(?, 4, 4)
    negative_loss = tf.reduce_mean(negative_loss)
    # tf.print(negative_loss)
    
    # 类别损失
    prob = tf.math.softmax(y_pred[..., 5:], axis=3) #(?, 4, 4, 20)  
    # tf.print(prob)  
    cls_loss = tf.square(y_true[..., 5:]-prob) #(?, 4, 4, 20)
    cls_loss = ob_exist * tf.reduce_sum(cls_loss, axis=3) # (?, 4, 4)
    cls_loss = tf.reduce_mean(cls_loss)
    # tf.print(cls_loss)

    total_loss = lambda_coord*loc_loss + lambda_coord*scale_loss + \
        positive_loss + lambda_noobj*negative_loss + cls_loss
    # tf.print(total_loss)
    return total_loss