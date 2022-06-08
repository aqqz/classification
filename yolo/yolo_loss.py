import tensorflow as tf
from data_gen import *
import numpy as np
from keras.losses import CategoricalCrossentropy
epr = 1e-6
def compute_iou(y_true, y_pred):
    # todo
    # [x, y, w, h]
    normal_x, normal_y, normal_w, normal_h = y_true[..., 0], y_true[..., 1], y_true[..., 2], y_true[..., 3] #(?, 4, 4)
    _normal_x, _normal_y, _normal_w, _normal_h = y_pred[..., 0], y_pred[..., 1], y_pred[..., 2], y_pred[..., 3] #(?, 4, 4)
    B = y_true.shape[0]
    
    offset_x = np.array([np.array([[i for i in range(S)]*S]).reshape(S, S)]*B).astype("float32") #(?, 4, 4)
    offset_x = tf.convert_to_tensor(offset_x)
    offset_y = np.array([np.array([[i for i in range(S)]*S]).reshape(S, S).transpose()]*B).astype("float32") #(?, 4, 4)
    offset_y = tf.convert_to_tensor(offset_y)
    
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
    # 获取标签、预测
    c = y_true[..., 0] #(?, 4, 4)
    _c = y_pred[..., 0] #(?, 4, 4)
    xywh = y_true[..., 1:5] #(?, 4, 4, 4)
    _xywh = y_pred[..., 1:5] #(?, 4, 4, 4)
    p = y_true[..., 5:] #(?, 4, 4, 20)
    _p = y_pred[..., 5:] #(?, 4, 4, 20)
    tf.print(_c)
    # tf.print(_xywh)
    # tf.print(_p)
    
    # 计算预测框和真值框的IOU
    iou_mask = compute_iou(xywh, _xywh) # (?, 4, 4)
    # tf.print(iou_mask)
    
    # 正样本中心点定位损失
    loc_loss = c*(tf.square(xywh[..., 0]-_xywh[..., 0]) + tf.square(xywh[..., 1]-_xywh[..., 1])) #(?, 4, 4)
    loc_loss = tf.reduce_mean(tf.reduce_sum(loc_loss, axis=[1, 2]))
    # tf.print(loc_loss)
    
    # 正样本宽高比损失
    scale_loss = c*(tf.square(tf.sqrt(xywh[..., 2]+epr)-tf.sqrt(_xywh[..., 2]+epr)) + \
        tf.square(tf.sqrt(xywh[..., 3]+epr)-tf.sqrt(_xywh[..., 3]+epr))) # (?, 4, 4)
    scale_loss = tf.reduce_mean(tf.reduce_sum(scale_loss, axis=[1, 2]))
    # tf.print(scale_loss)
    
    # 正样本网格分类损失
    positive_loss = c*tf.square(c*iou_mask-_c) #(?, 4, 4)
    positive_loss = tf.reduce_mean(tf.reduce_sum(positive_loss, axis=[1, 2]))
    # tf.print(positive_loss)

    # 负样本网格分类损失
    negative_loss = (1-c)*tf.square(c-_c) #(?, 4, 4)
    negative_loss = tf.reduce_mean(tf.reduce_sum(negative_loss, axis=[1, 2]))
    # tf.print(negative_loss)
    
    # 类别损失
    cls_loss = tf.square(p-_p) #(?, 4, 4, 20)
    cls_loss = c*tf.reduce_sum(cls_loss, axis=3) #(?, 4, 4)
    cls_loss = tf.reduce_mean(tf.reduce_sum(cls_loss, axis=[1, 2]))
    # loss_fn = CategoricalCrossentropy(axis=3)
    # cls_loss = loss_fn(p, _p) # (?, 4, 4)
    # tf.print(cls_loss)

    total_loss = lambda_coord*loc_loss + lambda_coord*scale_loss + positive_loss + \
        lambda_noobj*negative_loss + cls_loss
    # tf.print(total_loss)
    return tf.square(y_true-y_pred)