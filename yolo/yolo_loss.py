import tensorflow as tf
from data_gen import *
import numpy as np

epr = 1e-6
def compute_iou(y_true, y_pred):
    # todo
    # [x, y, w, h]
    # tf.print("y_true: ", y_true) #(?, 4, 4, 4)
    # tf.print("y_pred: ", y_pred) #(?, 4, 4, 4)
    normal_x, normal_y, normal_w, normal_h = y_true[..., 0], y_true[..., 1], y_true[..., 2], y_true[..., 3] #(?, 4, 4)
    _normal_x, _normal_y, _normal_w, _normal_h = y_pred[..., 0], y_pred[..., 1], y_pred[..., 2], y_pred[..., 3] #(?, 4, 4)
    B = y_true.shape[0]
    offset_x = np.array([np.array([[i for i in range(S)]*S]).reshape(S, S).transpose()]*B) #(?, 4, 4)
    offset_x = offset_x.astype("float32")
    offset_x = tf.convert_to_tensor(offset_x, dtype=tf.float32)
    # tf.print(offset_x)
    offset_y = np.array([np.array([[i for i in range(S)]*S]).reshape(S, S)]*B) #(?, 4, 4)
    offset_y = offset_y.astype("float32")
    offset_y = tf.convert_to_tensor(offset_y, dtype=tf.float32)
    # tf.print(offset_y)
    x = (normal_x + offset_x)*grid_size
    y = (normal_y + offset_y)*grid_size
    w = normal_w * img_size
    h = normal_h * img_size
    _x = (_normal_x + offset_x) *grid_size
    _y = (_normal_y + offset_y) *grid_size
    _w = _normal_w * img_size
    _h = _normal_h * img_size

    xmin = x - w/2
    ymin = y - h/2
    xmax = x + w/2
    ymax = y + h/2
    _xmin = _x - _w/2
    _ymin = _y - _h/2
    _xmax = _x + _w/2
    _ymax = _y + _h/2

    x_LU = tf.maximum(xmin, _xmin)
    y_LU = tf.maximum(ymin, _ymin)
    x_RD = tf.minimum(xmax, _xmax)
    y_RD = tf.maximum(ymax, _ymax)
    # compute batch grid ious
    w_inner = x_RD-x_LU
    w_inner = tf.where(tf.greater(w_inner, tf.zeros_like(w_inner)), w_inner, tf.zeros_like(w_inner))
    h_inner = y_RD-y_LU
    h_inner = tf.where(tf.greater(h_inner, tf.zeros_like(h_inner)), h_inner, tf.zeros_like(h_inner))
    
    S_inner = w_inner*h_inner
    S_label = (xmax-xmin)*(ymax-ymin)
    S_pred = (_xmax-_xmin)*(_ymax-_ymin)
    
    ious = S_inner / ((S_label + S_pred - S_inner) + epr)
    # tf.print(ious)
    return ious


def yolo_loss(y_true, y_pred):

    lambda_coord=5
    lambda_noobj=0.5
    

    # [batch, S, S, 5+C]
    # [:, :, :, ob, x, y, w, h, ...]
    # tf.print("y_true: ", y_true)
    # tf.print("y_pred: ", y_pred)
    ob_exist = y_true[..., 0] #(?, 4, 4)
    # tf.print(ob_exist)
    iou_mask = compute_iou(y_true[..., 1:5], y_pred[..., 1:5]) # (?, 4, 4)
    # tf.print(iou_mask)

    # 正样本中心点定位损失
    loc_loss = ob_exist*iou_mask*(tf.square(y_true[..., 1] - y_pred[..., 1]) + tf.square(y_true[..., 2] - y_pred[..., 2])) #(?, 4, 4)
    loc_loss = tf.reduce_mean(tf.reduce_sum(loc_loss, axis=[1, 2]))
    # tf.print(loc_loss)
    
    # 正样本宽高比损失
    scale_loss = ob_exist*iou_mask*(tf.square(tf.sqrt(y_true[..., 3]+epr)-tf.sqrt(y_pred[..., 3]+epr)) \
        + tf.square(tf.sqrt(y_true[..., 4]+epr)-tf.sqrt(y_pred[..., 4]+epr))) # (?, 4, 4)
    scale_loss = tf.reduce_mean(tf.reduce_sum(scale_loss, axis=[1, 2]))
    # tf.print(scale_loss)
    
    # 正样本网格分类损失
    positive_loss = ob_exist*iou_mask*tf.square(y_true[..., 0] - y_pred[..., 0]) #(?, 4, 4)
    positive_loss = tf.reduce_mean(tf.reduce_sum(positive_loss, axis=[1, 2]))
    # tf.print(positive_loss)

    # 负样本网格分类损失
    negative_loss = (1-ob_exist)*iou_mask*tf.square(y_true[..., 0] - y_pred[..., 0]) #(?, 4, 4)
    negative_loss = tf.reduce_mean(tf.reduce_sum(negative_loss, axis=[1, 2]))
    # tf.print(negative_loss)
    
    # 类别损失
    prob = tf.math.softmax(y_pred[..., 5:], axis=3) #(?, 4, 4, 20)  
    cls_loss = tf.square(y_true[..., 5:]-prob) #(?, 4, 4, 20)
    cls_loss = ob_exist * tf.reduce_sum(cls_loss, axis=3) # (?, 4, 4)
    cls_loss = tf.reduce_mean(tf.reduce_sum(cls_loss, axis=[1, 2]))
    # tf.print(cls_loss)

    total_loss = lambda_coord*loc_loss + lambda_coord*scale_loss + \
        positive_loss + lambda_noobj*negative_loss + cls_loss
    # tf.print(total_loss)
    return total_loss