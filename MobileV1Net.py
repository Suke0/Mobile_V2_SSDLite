#-- coding: utf-8 --

import tensorflow as tf

class MobileV1Net(tf.keras.Model):
    def __init__(self):
        super(MobileV1Net,self).__init__()
        #tf.keras.backend.set_learning_phase(True)
        self.conv_stage_0a = Conv_BN_ReLU(32, 3, 2)
        self.conv_stage_0b = DepthwiseConv_BN_ReLU_Conv_BN_ReLU(64,1)
        self.conv_stage_1a = DepthwiseConv_BN_ReLU_Conv_BN_ReLU(128,2)
        self.conv_stage_1b = DepthwiseConv_BN_ReLU_Conv_BN_ReLU(128,1)
        self.conv_stage_2a = DepthwiseConv_BN_ReLU_Conv_BN_ReLU(256,2)
        self.conv_stage_2b = DepthwiseConv_BN_ReLU_Conv_BN_ReLU(256,1)
        self.conv_stage_3a = DepthwiseConv_BN_ReLU_Conv_BN_ReLU(512,2)
        self.conv_stage_3b = DepthwiseConv_BN_ReLU_Conv_BN_ReLU(512,1)
        self.conv_stage_3c = DepthwiseConv_BN_ReLU_Conv_BN_ReLU(512,1)
        self.conv_stage_3d = DepthwiseConv_BN_ReLU_Conv_BN_ReLU(512,1)
        self.conv_stage_3e = DepthwiseConv_BN_ReLU_Conv_BN_ReLU(512,1)
        self.conv_stage_3f = DepthwiseConv_BN_ReLU_Conv_BN_ReLU(512,1)
        self.conv_stage_4a = DepthwiseConv_BN_ReLU_Conv_BN_ReLU(1024,2)
        self.conv_stage_4b = DepthwiseConv_BN_ReLU_Conv_BN_ReLU(1024,1)
        self.conv_stage_5a = tf.keras.layers.GlobalAveragePooling2D()
        self.conv_stage_5b = tf.keras.layers.Dense(1000)
        self.conv_stage_5c = tf.keras.layers.Softmax()
        pass

    def call(self,input):
        x = self.conv_stage_0a(input)  #输入(1, 224, 224, 3)
        x = self.conv_stage_0b(x)      #输入(1, 112, 112, 32)
        x = self.conv_stage_1a(x)      #输入(1, 112, 112, 64)
        x = self.conv_stage_1b(x)      #输入(1, 56, 56, 128)
        x = self.conv_stage_2a(x)      #输入(1, 56, 56, 128)
        x = self.conv_stage_2b(x)      #输入(1, 28, 28, 256)
        x = self.conv_stage_3a(x)      #输入(1, 28, 28, 256)
        x = self.conv_stage_3b(x)     #输入(1, 14, 14, 512)
        x = self.conv_stage_3c(x)     #输入(1, 14, 14, 512)
        x = self.conv_stage_3d(x)     #输入(1, 14, 14, 512)
        x = self.conv_stage_3e(x)     #输入(1, 14, 14, 512)
        x = self.conv_stage_3f(x)     #输入(1, 14, 14, 512)
        x = self.conv_stage_4a(x)      #输入(1, 14, 14, 512)
        x = self.conv_stage_4b(x)      #输入(1, 7, 7, 1024)
        x = self.conv_stage_5a(x)     #输入(1, 7, 7, 1024)
        x = self.conv_stage_5b(x)     #输入(1, 1024)
        x = self.conv_stage_5c(x)     #输入(1, 1000)
        return x
        pass
    pass

def DepthwiseConv_BN_ReLU_Conv_BN_ReLU(filters,strides):
    # kernel_size,
    # strides = (1, 1),
    # padding = 'valid',
    # depth_multiplier = 1,
    # data_format = None,
    # activation = None,
    # use_bias = True,
    # depthwise_initializer = 'glorot_uniform',
    # bias_initializer = 'zeros',
    # depthwise_regularizer = None,
    # bias_regularizer = None,
    # activity_regularizer = None,
    # depthwise_constraint = None,
    # bias_constraint = None,
    # ** kwargs
    return tf.keras.Sequential([
        tf.keras.layers.DepthwiseConv2D(3,strides=strides,padding='same',use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(filters,1,1,padding='same',use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
    ])
    pass

def Conv_BN_ReLU(filters,size,strides):

    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters,size,strides,padding='same',use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
    ])
    pass


if __name__=="__main__":
    import numpy as np
    input = np.random.random((1,224,224,3))
    model = MobileV1Net()

    res = model(input)
    print(model.summary())
    pass