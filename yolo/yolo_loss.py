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

    batch = y_true.shape[0]
    # offset
    offset_x = np.array([np.reshape(np.array([i for i in range(S)]*S), (S, S))]*batch)
    offset_x = tf.convert_to_tensor(offset_x, dtype=tf.float32) #(?, S, S)
    offset_y = np.array([np.reshape(np.array([i for i in range(S)]*S), (S, S)).transpose()]*batch)
    offset_y = tf.convert_to_tensor(offset_y, dtype=tf.float32) #(?, S, S)
    
    # box从相对于grid cell变为相对于整张图像比例
    box = tf.stack(
        [
            (xywh[..., 0] + offset_x) / S, 
            (xywh[..., 1] + offset_y) / S, 
            xywh[..., 2],
            xywh[..., 3],
        ], axis=-1
    )
    _box = tf.stack(
        [
            (_xywh[..., 0] + offset_x) / S, 
            (_xywh[..., 1] + offset_y) / S, 
            _xywh[..., 2],
            _xywh[..., 3],
        ], axis=-1
    )
    
    iou_score = compute_iou(box, _box) #(?, S, S)
    # tf.print(iou_score)

    object_mask = c
    noobject_mask = 1-object_mask
    positive_loss = object_mask*tf.square(iou_score-_c)
    positive_loss = tf.reduce_mean(tf.reduce_sum(positive_loss, axis=[1, 2]))
    # tf.print(positive_loss)
    negative_loss = noobject_mask*tf.square(0-_c)
    negative_loss = tf.reduce_mean(tf.reduce_sum(negative_loss, axis=[1, 2]))
    # tf.print(negative_loss)
    cls_loss = tf.square(p-_p) #(?, S, S, C)
    cls_loss = c*tf.reduce_sum(cls_loss, axis=3)
    cls_loss = tf.reduce_mean(tf.reduce_sum(cls_loss, axis=[1, 2]))
    # tf.print(cls_loss)
    loc_loss = object_mask*(tf.square(xywh[..., 0]-_xywh[..., 0]) + tf.square(xywh[..., 1]-_xywh[..., 1]))
    loc_loss = tf.reduce_mean(tf.reduce_sum(loc_loss, axis=[1, 2]))
    # tf.print(loc_loss)
    scale_loss = object_mask*(tf.square(tf.sqrt(xywh[..., 2]+1e-10)-tf.sqrt(_xywh[..., 2]+1e-10)) + \
        tf.square(tf.sqrt(xywh[..., 3]+1e-10)-tf.sqrt(_xywh[..., 3]+1e-10)))
    scale_loss = tf.reduce_mean(tf.reduce_sum(scale_loss, axis=[1, 2]))
    # tf.print(scale_loss)
    loss = positive_loss + 0.5*negative_loss + 2*cls_loss + 5*loc_loss + 5*scale_loss

    return loss


# if __name__ == '__main__':

#     box1 = np.expand_dims(np.array([0, 0, 0, 0], dtype="float32"), axis=0)
#     box2 = np.expand_dims(np.array([0, 0, 0, 0], dtype="float32"), axis=0)
#     print(box1)
#     print(box2)
#     print(compute_iou(box1, box2))

    # offset_x = np.array([np.reshape(np.array([i for i in range(S)]*S), (S, S))]*32)
    # offset_x = tf.convert_to_tensor(offset_x, dtype=tf.float32)
    # print(offset_x)
    # offset_y = np.array([np.reshape(np.array([i for i in range(S)]*S), (S, S)).transpose()]*32)
    # offset_y = tf.convert_to_tensor(offset_y, dtype=tf.float32)
    # print(offset_y)
    # pass