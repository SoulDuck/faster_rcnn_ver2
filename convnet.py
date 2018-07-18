from cnn import convolution2d
import tensorflow as tf
import numpy as np

def define_placeholder():
    x_ = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 1])
    im_dims = tf.placeholder(tf.int32, [None, 2])
    gt_boxes = tf.placeholder(tf.int32, [None, 5])
    phase_train = tf.placeholder(tf.bool)
    return x_ , im_dims , gt_boxes , phase_train

def simple_convnet(x_):
    print '###### Convolution Network building.... '
    print
    kernels=[5, 3, 3, 3, 3]
    out_channels=[16, 16, 32, 64, 128]
    strides = [2, 2, 1, 2 ,1 ]
    layer=x_
    for i in range(5):
        layer = convolution2d(name='conv_{}'.format(i), x=layer, out_ch=out_channels[i], k=kernels[i], s=strides[i],
                              padding='SAME')
    top_conv = tf.identity(layer , 'top_conv')
    _feat_stride = np.prod(strides)
    return top_conv , _feat_stride

def rpn_cls_layer(layer , n_anchors = 9 ):
    with tf.variable_scope('cls'):
        layer = convolution2d('rpn_cls_conv' ,layer, out_ch= n_anchors*2 , k=1 , act=None , s=1)
        layer = tf.identity(layer, name='cls_output')
        print '** cls layer shape : {}'.format(np.shape(layer)) #(1, ?, ?, 18)
    return layer
def rpn_bbox_layer(layer , n_anchors =9):
    with tf.variable_scope('bbox'):
        layer = convolution2d('rpn_bbox_conv' ,layer, out_ch= n_anchors*4 , k=1 , act=None , s=1)
        layer  = tf.identity(layer , name='cls_output')
        print '** cls layer shape : {}'.format(np.shape(layer)) #(1, ?, ?, 18)
    return layer

def sess_start():
    sess=tf.Session()
    init=tf.group(tf.global_variables_initializer() , tf.local_variables_initializer())
    sess.run(init)

    return sess

def rpn_cls_loss(rpn_cls_score , rpn_labels):
    shape = tf.shape(rpn_cls_score)
    rpn_cls_score_0 = tf.transpose(rpn_cls_score, [0, 3, 1, 2]) # (1, h, w, 18) ==>(1 , 18 , h , w)
    rpn_cls_score_1 = tf.reshape(rpn_cls_score_0, [shape[0], 2, shape[3] // 2 * shape[1], shape[2]])
    rpn_cls_score_2 = tf.transpose(rpn_cls_score_1, [0, 2, 3, 1])
    rpn_cls_score = tf.reshape(rpn_cls_score_2, [-1, 2])

    rpn_labels = tf.reshape(rpn_labels, [-1])

    cls_indices = tf.gather(rpn_cls_score, tf.where(tf.not_equal(rpn_labels, -1)), name='cls_indices')
    lab_indices = tf.gather(rpn_labels, tf.where(tf.not_equal(rpn_labels, -1)), name='lab_indices')
    rpn_cls_score = tf.reshape(cls_indices, [-1, 2])
    rpn_labels = tf.reshape(lab_indices, [-1])


    # Cross Enttropy
    rpn_cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_labels))

    return rpn_cross_entropy ,rpn_cls_score , rpn_labels



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


def rpn_bbox_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_inside_weights, rpn_outside_weights):
    RPN_BBOX_LAMBDA = 10.0
    with tf.variable_scope('rpn_bbox_loss'):

        # Transposing
        rpn_bbox_targets = tf.transpose(rpn_bbox_targets, [0, 2, 3, 1])
        rpn_inside_weights = tf.transpose(rpn_inside_weights, [0, 2, 3, 1])
        rpn_outside_weights = tf.transpose(rpn_outside_weights, [0, 2, 3, 1])

        # Extract ROI
        shape = tf.shape(rpn_bbox_targets)
        rpn_bbox_targets = tf.transpose(rpn_bbox_targets, [0, 3, 1, 2])
        rpn_bbox_targets = tf.reshape(rpn_bbox_targets, [shape[0], 4, shape[3] // 4 * shape[1], shape[2]])
        rpn_bbox_targets = tf.transpose(rpn_bbox_targets, [0, 2, 3, 1])
        rpn_bbox_targets = tf.reshape(rpn_bbox_targets, [-1, 4])


        exit()
        """
        rpn_cls_score_0 = tf.transpose(rpn_cls_score, [0, 3, 1, 2])  # (1, h, w, 18) ==>
        rpn_cls_score_1 = tf.reshape(rpn_cls_score_0, [shape[0], 2, shape[3] // 2 * shape[1], shape[2]])
        rpn_cls_score_2 = tf.transpose(rpn_cls_score_1, [0, 2, 3, 1])
        rpn_cls_score = tf.reshape(rpn_cls_score_2, [-1, 2])
        """

        # How far off was the prediction?

        diff = tf.multiply(rpn_inside_weights, rpn_bbox_pred - rpn_bbox_targets)
        diff_sL1 = smoothL1(diff, 3.0)
        # Only count loss for positive anchors. Make sure it's a sum.

        # tf.multiply(rpn_outside_weights, diff_sL1) shape : ? ? ? 36
        rpn_bbox_reg = tf.reduce_sum(tf.multiply(rpn_outside_weights, diff_sL1))

        # Constant for weighting bounding box loss with classification loss
        rpn_bbox_reg = RPN_BBOX_LAMBDA * rpn_bbox_reg

    return rpn_bbox_reg

def optimizer(cost):
    lr=0.01
    train_op= tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)
    return train_op


if __name__== '__main__':
    x_, im_dims, gt_boxes, phase_train=define_placeholder()
    top_conv, _feat_stride = simple_convnet(x_)
    rpn_cls = rpn_cls_layer(top_conv)
    rpn_bbox = rpn_bbox_layer(top_conv)



