from turtle import pos
import tensorflow as tf
from data_gen import *

lambda_coord = 5
lambda_noobj = 0.5
eps = 1e-7

def compute_iou(gtbox, pdbox):

    # 获取标签和网络预测的中心坐标和预测框宽高
    normal_x = gtbox[..., 0]
    normal_y = gtbox[..., 1]
    normal_w = gtbox[..., 2]
    normal_h = gtbox[..., 3]
    _normal_x = pdbox[..., 0]
    _normal_y = pdbox[..., 1]
    _normal_w = pdbox[..., 2]
    _normal_h = pdbox[..., 3]

    # 计算偏移矩阵
    B = pdbox.shape[0]
    offset_x = np.array([np.array([[i for i in range(S)]*S]).reshape(S, S)]*B).astype("float32") #(?, S, S)
    offset_x = tf.convert_to_tensor(offset_x)
    offset_y = np.array([np.array([[i for i in range(S)]*S]).reshape(S, S).transpose()]*B).astype("float32") #(?, S, S)
    offset_y = tf.convert_to_tensor(offset_y)

    # 计算预测和真值坐标
    x = (normal_x+offset_x)*grid_size
    y = (normal_y+offset_y)*grid_size
    w = normal_w*img_size
    h = normal_h*img_size
    _x = (_normal_x+offset_x)*grid_size
    _y = (_normal_y+offset_y)*grid_size
    _w = _normal_w*img_size
    _h = _normal_h*img_size

    # 还原真值和预测坐标
    xmin = x - w/2
    ymin = y - h/2
    xmax = x + w/2
    ymax = y + h/2
    _xmin = _x - _w/2
    _ymin = _y - _h/2
    _xmax = _x + _w/2
    _ymax = _y + _h/2

    # 计算交集框面积
    x_LU = tf.maximum(xmin, _xmin)
    y_LU = tf.maximum(ymin, _ymin)
    x_RD = tf.minimum(xmax, _xmax)
    y_RD = tf.minimum(ymax, _ymax)

    inner_w = tf.maximum(x_RD-x_LU, 0)
    inner_h = tf.maximum(y_RD-y_LU, 0)
    S_inner = inner_w*inner_h

    # 计算batch IOU
    S_label = (xmax-xmin)*(ymax-ymin)
    S_pred = (_xmax-_xmin)*(_ymax-_ymin)
    ious = S_inner / (S_label+S_pred-S_inner+eps)

    return ious




def yolo_loss(y_true, y_pred):

    # 获取标签和预测
    c = y_true[..., 0] # (?, S, S)
    _c = y_pred[..., 0] #(?, S, S)
    xywh = y_true[..., 1:5] #(?, S, S, 4)
    _xywh = y_pred[..., 1:5] #(?, S, S, 4)
    p = y_true[..., 5:] #(?, S, S, 20)
    _p = y_pred[..., 5:] #(?, S, S, 20)
    
    # 计算gtbox和pdbox的iou
    iou_score = compute_iou(xywh, _xywh) #(?, S, S)
    response_mask = tf.ones_like(iou_score) #(?, S, S)
    obj_mask = c #(?, S, S)
    noobj_mask = 1-c*response_mask #(?, S, S)

    # 损失计算
    loc_loss = obj_mask*response_mask*(tf.square(xywh[..., 0]-_xywh[..., 0])+tf.square(xywh[..., 1]-_xywh[..., 1])) #(?, S, S)
    loc_loss = tf.reduce_mean(tf.reduce_sum(loc_loss, axis=[1, 2]))
    # tf.print(loc_loss)

    scale_loss = obj_mask*response_mask*(
        tf.square(tf.sqrt(xywh[..., 2]+eps)-tf.sqrt(_xywh[..., 2]+eps)) + \
        tf.square(tf.sqrt(xywh[..., 3]+eps)-tf.sqrt(_xywh[..., 3]+eps))
        ) #(?, S, S)
    scale_loss = tf.reduce_mean(tf.reduce_sum(scale_loss, axis=[1, 2]))
    # tf.print(scale_loss)

    positive_loss = obj_mask*response_mask*tf.square(iou_score-_c) #(?, S, S)
    positive_loss = tf.reduce_mean(tf.reduce_sum(positive_loss, axis=[1, 2]))
    # tf.print(positive_loss)

    negative_loss = noobj_mask*tf.square(0-_c)
    negative_loss = tf.reduce_mean(tf.reduce_sum(negative_loss, axis=[1, 2]))
    # tf.print(negative_loss)

    probility_loss = obj_mask*tf.reduce_sum(-p*tf.math.log(_p), axis=3) #(?, S, S)
    probility_loss = tf.reduce_mean(tf.reduce_sum(probility_loss, axis=[1, 2]))
    # tf.print(probility_loss)

    total_loss = lambda_coord * loc_loss + lambda_coord * scale_loss + \
                positive_loss + lambda_noobj * negative_loss + probility_loss

    # tf.print(total_loss)
    return total_loss