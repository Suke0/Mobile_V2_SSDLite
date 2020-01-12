#-- coding: utf-8 --
#在跟了batchnorm层的卷积层设置偏置是多此一举；
import tensorflow as tf
Model = tf.keras.models.Model
Input, Lambda, Activation, Conv2D, DepthwiseConv2D, Reshape, Concatenate, BatchNormalization, ReLU \
    = tf.keras.layers.Input, tf.keras.layers.Lambda, tf.keras.layers.Activation, tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D, tf.keras.layers.Reshape, tf.keras.layers.Concatenate, tf.keras.layers.BatchNormalization, tf.keras.layers.ReLU
K = tf.keras.backend
from bodynet_ import bodynet_


def predict_block(inputs, out_channel, sym, id):
    name = 'ssd_' + sym + '{}'.format(id)
    x = DepthwiseConv2D(kernel_size=3, strides=1,
                           activation=None, use_bias=False, padding='same', name=name + '_dw_conv')(inputs)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name=name + '_dw_bn')(x)
    x = ReLU(6., name=name + '_dw_relu')(x)

    x = Conv2D(out_channel, kernel_size=1, padding='same', use_bias=False,
                  activation=None, name=name + 'conv2')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name=name + 'conv2_bn')(x)
    return x


def M2SSD300Net_(n_boxes=[4,6,6,6,4,4],n_classes=81):
   
    ############################################################################
    # Build the network.
    ############################################################################

    x1 = Input(shape=(300, 300, 3))

    links = bodynet_(x1)

    link1_cls = predict_block(links[0], n_boxes[0] * n_classes, 'cls', 1)
    link2_cls = predict_block(links[1], n_boxes[1] * n_classes, 'cls', 2)
    link3_cls = predict_block(links[2], n_boxes[2] * n_classes, 'cls', 3)
    link4_cls = predict_block(links[3], n_boxes[3] * n_classes, 'cls', 4)
    link5_cls = predict_block(links[4], n_boxes[4] * n_classes, 'cls', 5)
    link6_cls = predict_block(links[5], n_boxes[5] * n_classes, 'cls', 6)

    link1_box = predict_block(links[0], n_boxes[0] * 4, 'box', 1)
    link2_box = predict_block(links[1], n_boxes[1] * 4, 'box', 2)
    link3_box = predict_block(links[2], n_boxes[2] * 4, 'box', 3)
    link4_box = predict_block(links[3], n_boxes[3] * 4, 'box', 4)
    link5_box = predict_block(links[4], n_boxes[4] * 4, 'box', 5)
    link6_box = predict_block(links[5], n_boxes[5] * 4, 'box', 6)

    # Reshape
    cls1_reshape = Reshape((-1, n_classes), name='ssd_cls1_reshape')(link1_cls)
    cls2_reshape = Reshape((-1, n_classes), name='ssd_cls2_reshape')(link2_cls)
    cls3_reshape = Reshape((-1, n_classes), name='ssd_cls3_reshape')(link3_cls)
    cls4_reshape = Reshape((-1, n_classes), name='ssd_cls4_reshape')(link4_cls)
    cls5_reshape = Reshape((-1, n_classes), name='ssd_cls5_reshape')(link5_cls)
    cls6_reshape = Reshape((-1, n_classes), name='ssd_cls6_reshape')(link6_cls)

    box1_reshape = Reshape((-1, 4), name='ssd_box1_reshape')(link1_box)
    box2_reshape = Reshape((-1, 4), name='ssd_box2_reshape')(link2_box)
    box3_reshape = Reshape((-1, 4), name='ssd_box3_reshape')(link3_box)
    box4_reshape = Reshape((-1, 4), name='ssd_box4_reshape')(link4_box)
    box5_reshape = Reshape((-1, 4), name='ssd_box5_reshape')(link5_box)
    box6_reshape = Reshape((-1, 4), name='ssd_box6_reshape')(link6_box)


    cls = Concatenate(axis=1, name='ssd_cls')(
        [cls1_reshape, cls2_reshape, cls3_reshape, cls4_reshape, cls5_reshape, cls6_reshape])

    box = Concatenate(axis=1, name='ssd_box')(
        [box1_reshape, box2_reshape, box3_reshape, box4_reshape, box5_reshape, box6_reshape])

    cls = Activation('softmax', name='ssd_mbox_conf_softmax')(cls)

    predictions = Concatenate(axis=2, name='ssd_predictions')([cls, box])

    model = Model(inputs=x1, outputs=predictions)
    
    return model