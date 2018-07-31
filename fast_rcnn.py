import configure as cfg
import tensorflow as tf
from cnn import affine ,dropout
import tensorflow as tf


def roi_pool(featureMaps, rois, im_dims):
    '''
    Regions of Interest (ROIs) from the Region Proposal Network (RPN) are
    formatted as:
    (image_id, x1, y1, x2, y2)
    Note: Since mini-batches are sampled from a single image, image_id = 0s
    '''
    with tf.variable_scope('roi_pool'):
        # Image that the ROI is taken from (minibatch of 1 means these will all be 0)
        box_ind = tf.cast(rois[:, 0], dtype=tf.int32 )
        # ROI box coordinates. Must be normalized and ordered to [y1, x1, y2, x2]
        boxes = rois[:, 1:]
        normalization = tf.cast(tf.stack([im_dims[:, 1], im_dims[:, 0], im_dims[:, 1], im_dims[:, 0]], axis=1),
                                dtype=tf.float32)
        boxes = tf.div(boxes, normalization)
        boxes = tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=1)  # y1, x1, y2, x2

        # ROI pool output size
        crop_size = tf.constant([14, 14])
        # ROI pool
        pooledFeatures = tf.image.crop_and_resize(image=featureMaps, boxes=boxes, box_ind=box_ind, crop_size=crop_size)

        # Max pool to (7x7)
        pooledFeatures = tf.nn.max_pool(pooledFeatures, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    return pooledFeatures , boxes , box_ind

def fast_rcnn(top_conv , rois , im_dims , eval_mode ,num_classes , phase_train):
    print '###### Fast R-CNN building.... '
    print
    with tf.variable_scope('fast_rcnn'):
        keep_prob = cfg.FRCNN_DROPOUT_KEEP_RATE if eval_mode is False else 1.0
        pooledFeatures, boxes, box_ind = roi_pool(top_conv, rois, im_dims)# roi pooling
        layer = pooledFeatures  # ? 7,7 128 Same Output
        # print layer
        for i in range(len(cfg.FRCNN_FC_HIDDEN)):
            layer = affine('fc_{}'.format(i), layer, cfg.FRCNN_FC_HIDDEN[i])
            layer = dropout(layer, phase_train=phase_train, keep_prob=keep_prob)
        with tf.variable_scope('cls'):
            fast_rcnn_cls_logits = affine('cls_logits', layer, num_classes, activation=None)
        with tf.variable_scope('bbox'):
            fast_rcnn_bbox_logits = affine('bbox_logits', layer, num_classes * 4, activation=None)

    return fast_rcnn_cls_logits , fast_rcnn_bbox_logits




