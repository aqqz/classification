from tkinter import Y
import tensorflow as tf
from data_gen import *

l_coord = 5
l_noobj = 0.5
eps = 1e-10

# 损失函数和loss计算部分参考：
#  https://github.com/TowardsNorth/yolo_v1_tensorflow_guiyu/blob/991acae222a1754974c28df613bf3e2f7a45a3f2/yolo/yolo_net.py
def compute_iou(boxes1, boxes2):
    boxes1_t = tf.stack(
        [
            boxes1[..., 0] - boxes1[..., 2] / 2.0,
            boxes1[..., 1] - boxes1[..., 3] / 2.0,
            boxes1[..., 0] + boxes1[..., 2] / 2.0,
            boxes1[..., 1] + boxes1[..., 3] / 2.0,
        ], axis=-1
    )

    boxes2_t = tf.stack(
        [
            boxes2[..., 0] - boxes2[..., 2] / 2.0,
            boxes2[..., 1] - boxes2[..., 3] / 2.0,
            boxes2[..., 0] + boxes2[..., 2] / 2.0,
            boxes2[..., 1] + boxes2[..., 3] / 2.0,
        ], axis=-1
    )
    
    lu = tf.maximum(boxes1_t[..., :2], boxes2_t[..., :2])
    rd = tf.minimum(boxes1_t[..., 2:], boxes2_t[..., 2:])

    intersection = tf.maximum(rd-lu, 0)
    inter_square = intersection[..., 0]*intersection[..., 1]
    square1 = boxes1[..., 2]*boxes1[..., 3]
    square2 = boxes2[..., 2]*boxes2[..., 3]

    union_square = tf.maximum(square1+square2-inter_square, eps)
    return tf.clip_by_value(inter_square/union_square, 0.0, 1.0)



def yolo_loss(y_true, y_pred):

    batch = y_pred.shape[0]

    pred_confidence = tf.reshape(tf.stack([y_pred[..., 0], y_pred[..., 5]], axis=-1), [batch, S, S, B])
    pred_boxes = tf.reshape(tf.stack(
        [
            y_pred[..., 1],
            y_pred[..., 2],
            tf.sqrt(y_pred[..., 3] + eps),
            tf.sqrt(y_pred[..., 4] + eps),
            y_pred[..., 6],
            y_pred[..., 7],
            tf.sqrt(y_pred[..., 8] + eps),
            tf.sqrt(y_pred[..., 9] + eps), 
        ], axis=-1), [batch, S, S, B, 4])
    pred_classes = tf.reshape(y_pred[..., 10:], [batch, S, S, C])

    response = tf.reshape(y_true[..., 0], [batch, S, S, 1])
    boxes = tf.reshape(tf.stack(
        [
            y_true[..., 1],
            y_true[..., 2],
            tf.sqrt(y_true[..., 3] + eps),
            tf.sqrt(y_true[..., 4] + eps),
        ], axis=-1), [batch, S, S, 1, 4])
    boxes = tf.tile(boxes, [1, 1, 1, B, 1])
    classes = tf.reshape(y_true[..., 5:], [batch, S, S, C])

    offset = np.transpose(np.reshape(np.array([np.arange(S)]*S*B), (B, S, S)), (1, 2, 0))
    offset = tf.reshape(tf.constant(offset, dtype=tf.float32), [1, S, S, B])
    offset = tf.tile(offset, [batch, 1, 1, 1])
    offset_tran = tf.transpose(offset, (0, 2, 1, 3))

    predict_boxes_tran = tf.stack(
        [
            (pred_boxes[..., 0] + offset) / S,
            (pred_boxes[..., 1] + offset_tran) / S, 
            tf.square(pred_boxes[..., 2]),
            tf.square(pred_boxes[..., 3])
        ], axis=-1
    )
    boxes_tran = tf.stack(
        [
            (boxes[..., 0] + offset) / S, 
            (boxes[..., 1] + offset_tran) / S,
            tf.square(boxes[..., 2]),
            tf.square(boxes[..., 3])
        ], axis=-1
    )
    iou_score = compute_iou(boxes_tran, predict_boxes_tran) #(?, S, S, B)

    object_mask = tf.reduce_max(iou_score, 3, keepdims=True)
    object_mask = tf.cast((iou_score>=object_mask), tf.float32)*response

    noobject_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask
    
    # 类别损失
    class_delta = response*(pred_classes-classes)
    class_loss = tf.reduce_mean(
        tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3])
    )

    # 有目标的损失
    object_delta = object_mask*(pred_confidence-iou_score)
    object_loss = tf.reduce_mean(
        tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3])
    )

    # 没有目标的损失
    noobject_delta = noobject_mask*(pred_confidence-0)
    noobject_loss = tf.reduce_mean(
        tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3])
    )

    # 定位损失
    coord_mask = tf.expand_dims(object_mask, 4)
    coord_delta = coord_mask*(pred_boxes-boxes)
    coord_loss = tf.reduce_mean(
        tf.reduce_sum(tf.square(coord_delta), axis=[1, 2, 3, 4])
    )


    total_loss = l_coord*coord_loss + object_loss + l_noobj*noobject_loss + class_loss

    return total_loss