import tensorflow as tf
from data_gen import *

def compute_iou(box1, box2):
    # box1, box2: [x, y, w, h] (?, S, S, 4) in[0, 1]

    # box1_coord, box2_coord: [xmin, ymin, xmax, ymax] (?, S, S, 4)
    box1_coord = tf.stack([
        box1[..., 0] - box1[..., 2] / 2.0,
        box1[..., 1] - box1[..., 3] / 2.0,
        box1[..., 0] + box1[..., 2] / 2.0,
        box1[..., 1] + box1[..., 3] / 2.0,
    ], axis=-1)
    box2_coord = tf.stack([
        box2[..., 0] - box2[..., 2] / 2.0,
        box2[..., 1] - box2[..., 3] / 2.0,
        box2[..., 0] + box2[..., 2] / 2.0,
        box2[..., 1] + box2[..., 3] / 2.0,
    ], axis=-1)

    # inter: [xmin, ymin, xmax, ymax] (?, S, S, 4)
    inter = tf.stack([
        tf.maximum(box1_coord[..., 0], box2_coord[..., 0]),
        tf.maximum(box1_coord[..., 1], box2_coord[..., 1]), 
        tf.minimum(box1_coord[..., 2], box2_coord[..., 2]), 
        tf.minimum(box1_coord[..., 3], box2_coord[..., 3]),
    ], axis=-1)

    w_inter = tf.maximum(inter[..., 2]-inter[..., 0], 0)
    h_inter = tf.maximum(inter[..., 3]-inter[..., 1], 0)
    s_inter = w_inter*h_inter

    s_box1 = (box1_coord[..., 2]-box1_coord[..., 0])*(box1_coord[..., 3]-box1_coord[..., 1])
    s_box2 = (box2_coord[..., 2]-box2_coord[..., 0])*(box2_coord[..., 3]-box2_coord[..., 1])

    iou = s_inter / (s_box1 + s_box2 - s_inter + 1e-10)

    return iou



def yolo_loss(y_true, y_pred):

    c = y_true[..., 0] #(?, S, S)
    _c = y_pred[..., 0] #(?, S, S)
    xywh = y_true[..., 1:5] #(?, S, S, 4)
    _xywh = y_pred[..., 1:5] #(?, S, S, 4)
    p = y_true[..., 5:]
    _p = y_pred[..., 5:]
    # tf.print(_c)

    # offset
    
    iou_score = compute_iou(xywh, _xywh) #(?, S, S)
    tf.print(iou_score)

    object_mask = tf.where(tf.greater(iou_score, tf.ones_like(iou_score)*0.5), 1, 0)
    object_mask = tf.cast(object_mask, dtype=tf.float32)
    noobject_mask = 1-object_mask
    positive_loss = object_mask*tf.square(iou_score-_c)
    positive_loss = tf.reduce_mean(tf.reduce_sum(positive_loss, axis=[1, 2]))

    negative_loss = noobject_mask*tf.square(0-_c)
    negative_loss = tf.reduce_mean(tf.reduce_sum(negative_loss, axis=[1, 2]))

    cls_loss = tf.square(p-_p) #(?, S, S, C)
    cls_loss = tf.reduce_mean(c*tf.reduce_sum(cls_loss, axis=3))
    # todo: loc_loss, scale_loss
    
    loss = positive_loss + negative_loss + cls_loss

    return loss


if __name__ == '__main__':

    box1 = np.expand_dims(np.array([1, 1, 2, 2]), axis=0)
    box2 = np.expand_dims(np.array([2, 2, 2, 2]), axis=0)
    print(box1)
    print(box2)
    print(compute_iou(box1, box2))