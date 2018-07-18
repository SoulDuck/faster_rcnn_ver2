import tensorflow as tf
import numpy as np
from utils import next_img_gtboxes
from anchor_target_layer import anchor_target
from convnet import define_placeholder , simple_convnet , rpn_cls_layer , rpn_bbox_layer , sess_start , optimizer ,rpn_cls_loss , rpn_bbox_loss
import math
rpn_labels_op = tf.placeholder(dtype =tf.int32 , shape=[1,1,None,None])
rpn_bbox_targets_op = tf.placeholder(dtype =tf.float32 , shape=[1,36,None,None])
rpn_bbox_inside_weights_op = tf.placeholder(dtype =tf.float32 , shape=[1,36,None,None])
rpn_bbox_outside_weights_op = tf.placeholder(dtype =tf.float32 , shape=[1,36,None,None])
x_, im_dims, gt_boxes, phase_train = define_placeholder()
top_conv, _feat_stride = simple_convnet(x_)

# RPN CLS
rpn_cls = rpn_cls_layer(top_conv)
# RPN BBOX
rpn_bbox_pred = rpn_bbox_layer(top_conv)
# Loss
rpn_cls_loss_op ,A_op ,B_op = rpn_cls_loss(rpn_cls , rpn_labels_op)

rpn_bbox_loss_op = rpn_bbox_loss(rpn_bbox_pred ,rpn_bbox_targets_op , rpn_bbox_inside_weights_op , rpn_bbox_outside_weights_op , rpn_labels_op)
cost_op = rpn_bbox_loss_op
train_op = optimizer(cost_op)
sess=sess_start()

for i in range(2,100000):
    src_img , src_gt_boxes =next_img_gtboxes(i)
    h,w=np.shape(src_img)
    src_im_dims = [(h,w)]
    anchor_scales = [3,4,5]
    rpn_cls_score=np.zeros([1,int(math.ceil(h/8.)),int(math.ceil(w/8.)),512])
    rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights ,ori_labels= \
        anchor_target(rpn_cls_score=rpn_cls_score ,  gt_boxes = src_gt_boxes , im_dims = src_im_dims, _feat_stride= 8 ,anchor_scales=anchor_scales)
    src_img=src_img.reshape([1]+list(np.shape(src_img))+[1])
    feed_dict = {x_: src_img, im_dims: src_im_dims, gt_boxes: src_gt_boxes, phase_train: True,
                 rpn_labels_op: rpn_labels, rpn_bbox_targets_op: rpn_bbox_targets ,
                 rpn_bbox_inside_weights_op: rpn_bbox_inside_weights ,
                 rpn_bbox_outside_weights_op : rpn_bbox_outside_weights
                 }
    cost, _, rpn_cls_value ,A,B = sess.run( fetches = [cost_op , train_op , rpn_cls,A_op ,B_op], feed_dict = feed_dict)
    print cost

