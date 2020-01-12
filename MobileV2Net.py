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


def conv_block(inputs,filters,kernel,strides):
    x = tf.keras.layers.Conv2D(filters,kernel,strides=strides,padding="same",use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6.)(x)
    return x
    pass

def bottleneck(inputs,filters,kernel,t,alpha,s,r=False,expand=True):
    t_channel = tf.keras.backend.int_shape(inputs)[-1] * t
    c_channel = int(filters * alpha)
    if expand:
        x = conv_block(inputs,t_channel,1,1)
    else:
        x = inputs
    x = tf.keras.layers.DepthwiseConv2D(kernel,strides = s,padding='same',use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6.)(x)

    x = tf.keras.layers.Conv2D(c_channel,1,1,padding='same',use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    if r:
        x = tf.keras.layers.Add()([x,inputs])
    return x
    pass

def inverted_residual_block(inputs,filters,kernel,t,alpha,strides,n):
    x = bottleneck(inputs,filters,kernel,t,alpha,strides)
    for i in range(1,n):
        x = bottleneck(x,filters,kernel,t,alpha,1,True)
        pass
    return x
    pass

def MobileV2Net(input_shape,k,alpha=1.0):
    """MobileNetv2
    This function defines a MobileNetv2 architectures.
    # Arguments
        input_shape: An integer or tuple/list of 3 integers, shape
            of input tensor.
        k: Integer, number of classes.
        alpha: Integer, width multiplier, better in [0.35, 0.50, 0.75, 1.0, 1.3, 1.4].
    # Returns
        MobileNetv2 model.
    """
    inputs = tf.keras.layers.Input(shape=input_shape)
    first_filters = make_divisible(32*alpha,8)
    #stage_0
    x = conv_block(inputs,first_filters,3,strides=2)
    x = inverted_residual_block(x,16,3,t=1,alpha=alpha,strides=1,n=1,expand=False)
    # stage_1
    x = inverted_residual_block(x,24,3,t=6,alpha=alpha,strides=2,n=2)
    # stage_2
    x = inverted_residual_block(x,32,3,t=6,alpha=alpha,strides=2,n=3)
    # stage_3
    x = inverted_residual_block(x,64,3,t=6,alpha=alpha,strides=2,n=4)
    x = inverted_residual_block(x,96,3,t=6,alpha=alpha,strides=1,n=3)
    # stage_4
    x = inverted_residual_block(x,160,3,t=6,alpha=alpha,strides=2,n=3)
    x = inverted_residual_block(x,320,3,t=6,alpha=alpha,strides=1,n=1)

    if alpha > 1.0:
        last_filters = make_divisible(1280 * alpha, 8)
    else:
        last_filters =1280
        pass

    x = conv_block(x,last_filters,1,strides=1)

    #GlobalAveragePooling2D是平均池化的一个特例，它不需要指定pool_size和strides等参数，
    # 操作的实质是将输入特征图的每一个通道求平均得到一个数值。
    #GlobalAveragePooling2D最后返回的tensor是[batch_size, channels]两个维度的。

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Reshape((1,1,last_filters))(x)
    x = tf.keras.layers.Dropout(0.3,name = 'Dropout')(x)
    x = tf.keras.layers.Conv2D(k,1,padding="same")(x)

    x = tf.keras.layers.Softmax()(x)
    output = tf.keras.layers.Reshape((k,))(x)
    model = tf.keras.Model(inputs,output)
    return model
    pass

if __name__ == '__main__':
    import numpy as np
    model = MobileV2Net((224,224,3),100,1.0)
    vars = model.variables
    num = 0
    for i, v in enumerate(vars):
        num += np.prod(v.shape)
        print(v.name+"__"+str(v.shape))
    print(i+1)
    print(num)
    print("================================================================")
    print(model.summary())
    pass


