import configure as cfg
import tensorflow as tf
from cnn import affine ,dropout
import tensorflow as tf

def smoothL1(x, sigma):
    '''
    Tensorflow implementation of smooth L1 loss defined in Fast RCNN:
        (https://arxiv.org/pdf/1504.08083v2.pdf)

                    0.5 * (sigma * x)^2         if |x| < 1/sigma^2
    smoothL1(x) = {
                    |x| - 0.5/sigma^2           otherwise
    '''
    with tf.variable_scope('smoothL1'):
        conditional = tf.less(tf.abs(x), 1 / sigma ** 2)

        close = 0.5 * (sigma * x) ** 2
        far = tf.abs(x) - 0.5 / sigma ** 2

    return tf.where(conditional, close, far)

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
        FRCNN_DROPOUT_KEEP_RATE =0.5
        FRCNN_FC_HIDDEN = [1024, 1024]
        keep_prob = FRCNN_DROPOUT_KEEP_RATE if eval_mode is False else 1.0
        pooledFeatures, boxes, box_ind = roi_pool(top_conv, rois, im_dims)# roi pooling
        layer = pooledFeatures  # ? 7,7 128 Same Output
        # print layer
        for i in range(len(FRCNN_FC_HIDDEN)):
            layer = affine('fc_{}'.format(i), layer, FRCNN_FC_HIDDEN[i])
            layer = dropout(layer, phase_train=phase_train, keep_prob=keep_prob)
        with tf.variable_scope('cls'):
            fast_rcnn_cls_logits = affine('cls_logits', layer, num_classes, activation=None)
        with tf.variable_scope('bbox'):
            fast_rcnn_bbox_logits = affine('bbox_logits', layer, num_classes * 4, activation=None)

    return fast_rcnn_cls_logits , fast_rcnn_bbox_logits


def fast_rcnn_cls_loss(fast_rcnn_cls_score, labels):
    '''
    Calculate the fast RCNN classifier loss. Measures how well the fast RCNN is
    able to classify objects from the RPN.

    Standard cross-entropy loss on logits
    '''
    with tf.variable_scope('fast_rcnn_cls_loss'):
        # Cross entropy error
        fast_rcnn_cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fast_rcnn_cls_score, labels=labels))

    return fast_rcnn_cross_entropy


def fast_rcnn_bbox_loss(fast_rcnn_bbox_pred, bbox_targets, roi_inside_weights, roi_outside_weights):
    '''
    Calculate the fast RCNN bounding box refinement loss. Measures how well
    the fast RCNN is able to refine localization.
    lam/N_reg * sum_i(p_i^* * L_reg(t_i,t_i^*))
    lam: classification vs bbox loss balance parameter
    N_reg: Number of anchor locations (~2500)
    p_i^*: ground truth label for anchor (loss only for positive anchors)
    L_reg: smoothL1 loss
    t_i: Parameterized prediction of bounding box
    t_i^*: Parameterized ground truth of closest bounding box

    TODO: rpn_inside_weights likely deprecated; might consider obliterating
    '''
    with tf.variable_scope('fast_rcnn_bbox_loss'):
        FRCNN_BBOX_LAMBDA =1
        # How far off was the prediction?
        diff = tf.multiply(roi_inside_weights, fast_rcnn_bbox_pred - bbox_targets)
        diff_sL1 = smoothL1(diff, 1.0)
        # Only count loss for positive anchors
        roi_bbox_reg = tf.reduce_mean(tf.reduce_sum(tf.multiply(roi_outside_weights, diff_sL1), reduction_indices=[1]))
        # Constant for weighting bounding box loss with classification loss
        roi_bbox_reg = FRCNN_BBOX_LAMBDA * roi_bbox_reg

    return roi_bbox_reg




