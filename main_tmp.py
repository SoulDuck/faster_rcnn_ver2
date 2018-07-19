import tensorflow as tf
import numpy as np
from utils import next_img_gtboxes , draw_rectangles
from anchor_target_layer import anchor_target
from convnet import define_placeholder , simple_convnet , rpn_cls_layer , rpn_bbox_layer , sess_start , optimizer ,rpn_cls_loss , rpn_bbox_loss ,bbox_loss
import math
import roi
import sys
rpn_labels_op = tf.placeholder(dtype =tf.int32 , shape=[1,1,None,None])
rpn_bbox_targets_op = tf.placeholder(dtype =tf.float32 , shape=[1,36,None,None])
rpn_bbox_inside_weights_op = tf.placeholder(dtype =tf.float32 , shape=[1,36,None,None])
rpn_bbox_outside_weights_op = tf.placeholder(dtype =tf.float32 , shape=[1,36,None,None])

bbox_targets_op = tf.placeholder(dtype =tf.float32 , shape=[None,4])
bbox_inside_weights_op = tf.placeholder(dtype =tf.float32 , shape=[None,4])
bbox_outside_weights_op = tf.placeholder(dtype =tf.float32 , shape=[None,4])

x_, im_dims, gt_boxes, phase_train = define_placeholder()
top_conv, _feat_stride = simple_convnet(x_)

# RPN CLS
rpn_cls = rpn_cls_layer(top_conv)
# RPN BBOX
rpn_bbox_pred = rpn_bbox_layer(top_conv)
# CLS LOSS
rpn_cls_loss_op ,A_op ,B_op = rpn_cls_loss(rpn_cls , rpn_labels_op)

# BBOX LOSS
# C_op : indiced rpn bbox  pred op
# D_op :indiced rpn target  op
# E_op : rpn_inside_weights
# F_op : indices
rpn_bbox_loss_op , diff_op , C_op , D_op ,E_op ,F_op= \
    bbox_loss(rpn_bbox_pred ,bbox_targets_op , bbox_inside_weights_op , bbox_outside_weights_op , rpn_labels_op)

anchor_scales = [3,4,5]
blobs_op, scores_op= roi.roi_proposal(rpn_cls , rpn_bbox_pred , im_dims , _feat_stride , anchor_scales ,is_training=True)
cost_op = rpn_cls_loss_op + rpn_bbox_loss_op
train_op = optimizer(cost_op)
sess=sess_start()

max_iter = 100000
for i in range(2,100000):
    src_img , src_gt_boxes =next_img_gtboxes(i)
    h,w=np.shape(src_img)
    src_im_dims = [(h,w)]

    rpn_cls_score=np.zeros([1,int(math.ceil(h/8.)),int(math.ceil(w/8.)),512])
    rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights, bbox_targets, bbox_inside_weights, bbox_outside_weights= anchor_target(
        rpn_cls_score=rpn_cls_score, gt_boxes=src_gt_boxes, im_dims=src_im_dims, _feat_stride=8, anchor_scales=anchor_scales)

    # utils here
    src_img=src_img.reshape([1]+list(np.shape(src_img))+[1])
    feed_dict = {x_: src_img, im_dims: src_im_dims, gt_boxes: src_gt_boxes, phase_train: True,
                 rpn_labels_op: rpn_labels,
                 bbox_targets_op: bbox_targets ,
                 bbox_inside_weights_op: bbox_inside_weights ,
                 bbox_outside_weights_op : bbox_outside_weights
                 }

    cost, _ ,rpn_cls_value, A, B, diff, C, D ,E ,F , blobs ,scores= sess.run(
        fetches=[cost_op, train_op, rpn_cls, A_op, B_op, diff_op, C_op, D_op ,E_op ,F_op,blobs_op ,scores_op], feed_dict=feed_dict)

    pos_blobs=blobs[np.where([scores > 0.5])[1]]

    if i % 100 ==0:
        print '\t', cost
        print len(pos_blobs)
        savepath = './result/{}.png'.format(i)
        src_img=np.squeeze(src_img)
        draw_rectangles(src_img ,pos_blobs, savepath)

    sys.stdout.write('\r Progress {} {}'.format(i,max_iter))
    sys.stdout.flush()


