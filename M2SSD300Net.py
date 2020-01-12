#-- coding: utf-8 --
import collections
import tensorflow as tf
import numpy as np
from bodynet import body_net
from SSDUtil import get_encode_anchor_for_layer,get_anchor_sizes,get_anchors_wh

def detect_block(inputs,out_channel):
    x = tf.keras.layers.DepthwiseConv2D(3,1,padding='same',use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    x = tf.keras.layers.Conv2D(out_channel,1,1,padding='same',use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x
    pass

# M2SSDNetParams = collections.namedtuple('M2SSDNetParams',
#                                         ['n_classes',
#                                          'confidence_thresh',
#                                          'iou_threshold',
#                                          'variances',
#                                          'min_scale',
#                                          'max_scale',
#                                          'max_n_anchors',
#                                          'anchor_ratios'])

def M2SSDNet(inputs_shape,net_params):
    inputs = tf.keras.layers.Input(shape=inputs_shape[1:],batch_size=inputs_shape[0])
    stage_results = body_net(inputs)
    anchor_sizes = get_anchor_sizes(net_params.min_scale, net_params.max_scale,net_params.max_n_anchors)
    anchors = get_anchors_wh(anchor_sizes,net_params.anchor_ratios)

    # num_anchors = len(sizes) + len(ratios)
    loc_preds, cls_preds = [], []
    for i, res in enumerate(stage_results):
        pre_shape = res.get_shape().as_list()[0:-1]
        n_anchors = len(anchor_sizes[i]) + len(net_params.anchor_ratios[i])

        loc_pre_shape = pre_shape + [n_anchors, 4]
        # shape=(batch_size,img_w,img_h,n_anchors,4)
        loc_pre_shape_ = loc_pre_shape
        loc_pre_shape = (loc_pre_shape[0],np.prod(loc_pre_shape[1:4]),loc_pre_shape[4]) #(batch_size,n_boxes,4)

        cls_pre_shape = pre_shape + [n_anchors, net_params.n_classes]
        cls_pre_shape = (cls_pre_shape[0],np.prod(cls_pre_shape[1:4]),cls_pre_shape[4])#(batch_size,n_boxes,n_classes)
        # numbers of anchors
        # n_anchors = len(self.sizes) + len(self.ratios)
        # location predictions
        loc_pred = detect_block(stage_results[i],(len(anchor_sizes[i]) + len(net_params.anchor_ratios[i])) * 4)
        loc_pred = tf.reshape(loc_pred, loc_pre_shape) #(batch_size,n_boxes,4)
        #加上anchors和variances
        encoded_anchors = get_encode_anchor_for_layer(loc_pre_shape_,anchors[i]) #(batch_size,n_boxes,8)
        loc_pred = tf.concat([loc_pred,encoded_anchors],-1) #(batch_size,n_boxes,12)
        loc_preds.append(loc_pred)
        # class prediction
        cls_pred = detect_block(stage_results[i],(len(anchor_sizes[i]) + len(net_params.anchor_ratios[i])) * net_params.n_classes)
        cls_pred = tf.reshape(cls_pred, cls_pre_shape)
        cls_pred = tf.nn.softmax(cls_pred)
        cls_preds.append(cls_pred)
        pass

    loc_preds_ = tf.concat(loc_preds,1)
    cls_preds_ = tf.concat(cls_preds,1)
    predictions = tf.concat([cls_preds_, loc_preds_], -1)#(batch_size,2268,n_classes+4+4+4)

    return tf.keras.Model(inputs, predictions)
    pass