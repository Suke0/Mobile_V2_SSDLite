#-- coding: utf-8 --

import numpy as np
import tensorflow as tf
import collections
import os
import cv2
import glob
from M2SSD300Net import M2SSDNet
from M2SSD300Net_ import M2SSD300Net_
from M2SSD300Loss import SSDLoss
from BatchGenerator import BatchGenerator
from SSDUtil import non_max_suppression,draw_boxes

# 定义分类
from weights import WeightReader

LABELS = ["backgroud",
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

LABELS = ["backgroud",
        'person','bicycle','car','motorbike','aeroplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter',
        'bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase',
        'frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle',
        'wine glass','cup', 'fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut',
        'cake','chair','sofa','pottedplant','bed','diningtable','toilet','tvmonitor','laptop','mouse','remote','keyboard','cell phone',
        'microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
        ]


params = collections.namedtuple('M2SSDNetParams',
                                        ['n_classes',
                                         'confidence_thresh',
                                         'iou_threshold',
                                         'variances',
                                         'min_scale',
                                         'max_scale',
                                         'max_n_anchors',
                                         'anchor_ratios'])

#LABELS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
params.n_classes = len(LABELS)
params.confidence_thresh = 0.5
params.iou_threshold = 0.45
params.variances = [0.1,0.1,0.2,0.2]
params.min_scale = 0.1
params.max_scale = 0.9
params.max_n_anchors = 6
params.anchor_ratios = [ [2, .5],
                                 [2, .5, 3, 1./3],
                                 [2, .5, 3, 1./3],
                                 [2, .5, 3, 1./3],
                                 [2, .5],
                                 [2, .5] ]

# 获取当前目录
PROJECT_ROOT = os.path.dirname(__file__)

# 定义样本路径
train_ann_dir = os.path.join(PROJECT_ROOT, "voc_train_data", "ann", "*.xml")
train_img_dir = os.path.join(PROJECT_ROOT, "voc_train_data", "img")

ann_dir = os.path.join(PROJECT_ROOT, "data", "ann", "*.xml")
img_dir = os.path.join(PROJECT_ROOT, "data", "img")

val_ann_dir = os.path.join(PROJECT_ROOT, "voc_val_data", "ann", "*.xml")
val_img_dir = os.path.join(PROJECT_ROOT, "voc_val_data", "img")

test_img_file = os.path.join(PROJECT_ROOT, "voc_test_data", "val_img","*")
train_test_img_file = os.path.join(PROJECT_ROOT, "voc_train_data", "img","*")
batch_size = 64

iou_threshold = 0.5
# 获取该路径下的xml
train_ann_fnames = glob.glob(train_ann_dir)
ann_fnames = glob.glob(ann_dir)
val_ann_fnames = glob.glob(val_ann_dir)
test_img_fnames = glob.glob(test_img_file)
log_dir =os.path.join(PROJECT_ROOT, "log")
def lr_schedule(epoch):
    if epoch < 10:
        return 0.0001
    elif epoch < 20:
        return 0.00001
    else:
        return 0.00001

class TrainCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        pass
    pass


def train(model):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),
                  loss= SSDLoss().compute_loss,
                  metrics = None)

    print(model.summary())

    train_data_generator = BatchGenerator(train_ann_fnames,train_img_dir,LABELS,batch_size,params.anchor_ratios,shuffle = True)
    val_data_generator = BatchGenerator(val_ann_fnames,val_img_dir,LABELS,batch_size,params.anchor_ratios,shuffle = False)

    callbacks = [#TrainCallback(),
                #tf.keras.callbacks.LearningRateScheduler(schedule=lr_schedule, verbose=1),
                 #tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False),
                 tf.keras.callbacks.ModelCheckpoint(
                     os.path.join(log_dir, "ssd_voc_{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5"),
                     monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=True,save_freq='epoch')]

    model.fit_generator(train_data_generator, epochs=120, steps_per_epoch=500,
                        callbacks=callbacks, validation_data=val_data_generator,initial_epoch=0)
    pass


def detect(model):

    for i, img_fname in enumerate(test_img_fnames):
        img = cv2.imread(img_fname)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        preprocess_img = cv2.resize(img / 255., (300,300))
        new_image = np.expand_dims(preprocess_img, axis=0)

        outputs = model.predict(new_image)
        predictions = np.squeeze(outputs,0) #(n_boxes,n_classes+4+4+4)
        result = non_max_suppression(predictions,confidence_threshold=0.5, iou_threshold=0.4)
        if len(result) == 0:
            continue
        draw_boxes(i,result,img_fname,LABELS,(300,300))
        pass
    pass

def validate_sample():
    generator = BatchGenerator(train_ann_fnames, train_img_dir, LABELS, batch_size, params.anchor_ratios, shuffle=False)
    train_test_img_fnames = glob.glob(train_test_img_file)
    for i,img_fname in enumerate(train_test_img_fnames):
        _, y_encoded = generator.get(i)
        result = non_max_suppression(y_encoded, confidence_threshold=0.5, iou_threshold=0.4)
        if len(result) > 0:
            #draw_boxes(i, result, img_fname, LABELS, (300, 300))
            pass
    pass
    pass

if __name__ ==  '__main__':
    weights_path = os.path.join(log_dir, "ssd_voc_01_loss-2.9313_val_loss-5.5237.h5")
    model = M2SSDNet((1, 300, 300, 3), params)
    print(len(model.variables))
    n_variable = sum(map(lambda v: np.prod(v.shape),model.variables))
    print(n_variable)
    #model.load_weights(weights_path, by_name=True)
    wr = WeightReader(weights_path)
    print(model.summary())
    print("=================================================")
    model1 = M2SSD300Net_()
    # for v, v1 in zip(model.variables, model1.variables):
    #     print(v.name+"_"+str(v.shape))
    #     print(v1.name+"_"+str(v1.shape))
    #     pass
    print("=================================================")
    print(len(model1.variables))
    n_variable = sum(map(lambda v: np.prod(v.shape), model1.variables))
    print(n_variable)
    print(model1.summary())
    # i = 0
    # for v in model.variables:
    #     if i<5:
    #         print(v)
    #         pass
    #     i+=1
    #     pass
    #train(model)
    #detect(model)

    #validate_sample()
    pass

