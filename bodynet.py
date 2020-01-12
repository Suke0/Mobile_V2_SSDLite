#-- coding: utf-8 --
#在跟了batchnorm层的卷积层设置偏置是多此一举；
import tensorflow as tf


def make_divisible(v,divisor,min_value=None):
    if min_value is None:
        min_value = divisor
        pass
    new_v = max(min_value, int(v+divisor/2)//divisor*divisor)
    #make sure that round down does not go down by more than 10%
    if new_v < 0.9 * v:
        new_v += divisor
        pass
    return new_v
    pass


def correct_padding(shape, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.
    # Arguments
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.
    # Returns
        A tuple.
    """
    input_size = shape[1:3]
    adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
    correct = (kernel_size // 2, kernel_size // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))
    pass


def conv_block(inputs,filters,kernel_size,strides):
    x = tf.keras.layers.Conv2D(filters,kernel_size,strides=strides,padding="same",use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6.)(x)
    return x
    pass


def bottleneck(inputs,filters,alpha=1.0):
    t = 6
    t_channel = tf.keras.backend.int_shape(inputs)[-1] * t
    c_channel = int(filters * alpha)

    x = conv_block(inputs,t_channel,1,1)

    x = tf.keras.layers.DepthwiseConv2D(3,strides=1,padding='same',use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6.)(x)

    x = tf.keras.layers.Conv2D(c_channel,1,strides=1,padding='same',use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Add()([x,inputs])
    return x
    pass


def inverted_res_block(inputs,filters,strides=1,t=1,alpha=1.0,n=1,expand=True,two_outputs=False):
    t_channel = inputs.shape[-1] * t
    c_channel = int(filters * alpha)
    if expand:
        x = conv_block(inputs, t_channel, 1, 1)
        res = x
    else:
        x = inputs
    padding = 'same'
    if strides == 2:
        x = tf.keras.layers.ZeroPadding2D(padding=correct_padding(x.shape, 3))(x)
        padding = 'valid'
    x = tf.keras.layers.DepthwiseConv2D(3, strides=strides, padding=padding, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6.)(x)

    x = tf.keras.layers.Conv2D(c_channel, 1, 1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    for i in range(1,n):
        x = bottleneck(x,filters,alpha=alpha)
        pass
    if two_outputs:
        return x, res
        pass
    return x
    pass


def followed_down_sample_block(inputs,in_filters,out_filters):
    x = tf.keras.layers.Conv2D(in_filters,1,strides=1,padding="same",use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6.)(x)

    x = tf.keras.layers.ZeroPadding2D(padding=correct_padding(x.shape,3))(x)
    x = tf.keras.layers.DepthwiseConv2D(3, strides=2, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6.)(x)

    x = tf.keras.layers.Conv2D(out_filters, 1, strides=1, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6.)(x)

    return x
    pass


#96*19*19，1280*10*10  512*5*5  256*3*3  256*3*3 128*1*1
def body_net(input_array):
    alpha = 1.0
    first_filters = make_divisible(32*alpha,8)
    #stage_0
    x = conv_block(input_array,first_filters,3,strides=2)
    x = inverted_res_block(x,16,t=1,alpha=alpha,strides=1,n=1,expand=False)
    # stage_1
    x = inverted_res_block(x,24,t=6,alpha=alpha,strides=2,n=2)
    # stage_2
    x = inverted_res_block(x,32,t=6,alpha=alpha,strides=2,n=3)
    # stage_3
    x = inverted_res_block(x,64,t=6,alpha=alpha,strides=2,n=4)
    x = inverted_res_block(x,96,t=6,alpha=alpha,strides=1,n=3)
    # stage_4
    x,res4 = inverted_res_block(x,160,t=6,alpha=alpha,strides=2,n=3,two_outputs=True)
    x = inverted_res_block(x,320,t=6,alpha=alpha,strides=1,n=1)

    if alpha > 1.0:
        last_filters = make_divisible(1280 * alpha, 8)
    else:
        last_filters =1280
        pass
    # stage_5
    x = conv_block(x,last_filters,1,strides=1)
    res5 = x
    # stage_6
    x = followed_down_sample_block(x,256,512)
    res6 = x
    # stage_7
    x = followed_down_sample_block(x, 128, 256)
    res7 = x
    # stage_8
    x = followed_down_sample_block(x, 128, 256)
    res8 = x
    # stage_9
    x = followed_down_sample_block(x, 64, 128)
    res9 = x
    return [res4, res5, res6, res7, res8, res9]
    # (1, 19, 19, 576)
    # (1, 10, 10, 1280)
    # (1, 5, 5, 512)
    # (1, 3, 3, 256)
    # (1, 2, 2, 256)
    # (1, 1, 1, 128)
    pass

if __name__ == '__main__':
    import numpy as np
    inputs = np.random.random((1,300,300,3))
    res = body_net(inputs)
    print(res[0].shape)
    print(res[1].shape)
    print(res[2].shape)
    print(res[3].shape)
    print(res[4].shape)
    print(res[5].shape)

    pass


