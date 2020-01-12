#-- coding: utf-8 --

import numpy as np
import math
import os
import cv2
import tensorflow as tf
from xml.etree.ElementTree import parse
from random import shuffle
from SSDUtil import get_encode_anchor_for_layer, get_anchors_wh, get_anchor_sizes, encode_box,resize_image,match_max_iou,match_many

class BatchGenerator(tf.keras.utils.Sequence):
    def __init__(self,ann_fnames,img_dir,label_names,batch_size,anchor_ratios,
                 min_scale=0.1,max_scale=0.9, feature_map_sizes=[19, 10, 5, 3, 2, 1],pos_iou_threshold=0.5,
                 neg_iou_limit=0.3,
                 net_size=300,jitter=True,shuffle=True):
        self.ann_fnames= ann_fnames
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.n_classes = len(label_names)
        self.pos_iou_threshold = pos_iou_threshold
        self.neg_iou_limit = neg_iou_limit
        self.feature_map_sizes = feature_map_sizes

        anchor_sizes = get_anchor_sizes(min_scale,max_scale)

        self.anchors_wh = get_anchors_wh(anchor_sizes, anchor_ratios)
        self.jitter = jitter
        self.shuffle = shuffle
        self.label_names = label_names
        self.net_size = net_size
        pass


    def __len__(self):
        return math.ceil(len(self.ann_fnames) / self.batch_size)

    def __getitem__(self, idx):
        xs, y_encodeds = [], []
        for i in range(self.batch_size):
            x, y_encoded = self.get(idx * self.batch_size + i)
            xs.append(x)
            y_encodeds.append(y_encoded)
            pass
        return np.array(xs).astype(np.float32), np.array(y_encodeds).astype(np.float32)
        pass

    def on_epoch_end(self):
        if self.shuffle:
            shuffle(self.ann_fnames)
            pass
        pass
    # 初始化标签
    def generate_encoding_template(self):
        y_templates = []
        for i in range(len(self.feature_map_sizes)):
            #(n_boxes,4+4)
            encode_anchors_for_layer = get_encode_anchor_for_layer((1,self.feature_map_sizes[i],self.feature_map_sizes[i],len(self.anchors_wh[i]),2),self.anchors_wh[i])
            y_empty_for_layer = np.zeros(np.append(encode_anchors_for_layer.shape[0:-1],self.n_classes + 4),dtype=np.float32)
            y_template = np.concatenate([y_empty_for_layer,encode_anchors_for_layer],-1)
            y_template = np.squeeze(y_template)
            y_templates.append(y_template)
            pass

        y_templates = np.concatenate(y_templates,0)
        return y_templates #(n_boxes,n_classes+4+4+4)
        pass


    def get(self,index):
        net_size = self.net_size
        #解析标注文件
        fname, boxes, coded_labels = parse_annotation(self.ann_fnames[index],self.img_dir,self.label_names)
        #读取图片，并按照设置修改图片尺寸

        image = cv2.imread(fname) #返回（高度，宽度，通道数）的元组
        boxes_ = np.copy(boxes)
        if self.jitter:#是否要增强数据
            #image, boxes_ = make_jitter_on_image(image,boxes_)
            pass
        image, ground_truth_boxes = resize_image(image,boxes_,net_size,net_size) #对原始图片进行缩放，并且重新计算True box的位置坐标
        #boxes_为[x1,y1,x2,y2]

        y_encode_template = self.generate_encoding_template() #(n_boxes,n_classes+4+4+4)
        y_encode_template[:,0] = 1
        iou_matrix = iou(ground_truth_boxes,y_encode_template[:,-8:-4]) #(n_ground_truth_boxes, n_boxes)

        match_max_iou(iou_matrix, y_encode_template, np.copy(ground_truth_boxes), np.copy(coded_labels))

        # 匹配原则3：将与任一真实框的iou值位于[neg_iou_limit,pos_iou_threshold]之间的anchor_tensor舍弃
        iou_matrix_t = iou_matrix.T
        for idx, item in enumerate(iou_matrix_t):
            if np.all(item < self.pos_iou_threshold) and np.all(item > self.neg_iou_limit):
                if y_encode_template[idx, 0] == 1:
                    y_encode_template[idx, 0:-12] = 0
                pass
            pass

        match_many(iou_matrix, y_encode_template, np.copy(ground_truth_boxes), np.copy(coded_labels),self.pos_iou_threshold)
        print("================================================")
        print(f"正例样本数：{np.sum(y_encode_template[:,1:-12])}")
        print("================================================")

        encode_box(y_encode_template)
        #print(f"n_pos:{n_pos}")
        #print(f"ignore_neg:{ignore_neg}")
        return image/255.,y_encode_template
        pass
    pass


class PascalVocXmlParser(object):
    def __init__(self):
        pass

    def get_fname(self, annotation_file):
        root = self._root_tag(annotation_file)
        return root.find("filename").text

    def get_width(self, annotation_file):
        tree = self._tree(annotation_file)
        for elem in tree.iter():
            if 'width' in elem.tag:
                return int(elem.text)

    def get_height(self, annotation_file):
        tree = self._tree(annotation_file)
        for elem in tree.iter():
            if 'height' in elem.tag:
                return int(elem.text)

    def get_labels(self, annotation_file):
        root = self._root_tag(annotation_file)
        labels = []
        obj_tags = root.findall("object")
        for t in obj_tags:
            labels.append(t.find("name").text)
        return labels

    def get_boxes(self, annotation_file):
        root = self._root_tag(annotation_file)
        bbs = []
        obj_tags = root.findall("object")
        for t in obj_tags:
            box_tag = t.find("bndbox")
            x1 = box_tag.find("xmin").text
            y1 = box_tag.find("ymin").text
            x2 = box_tag.find("xmax").text
            y2 = box_tag.find("ymax").text
            box = np.array([int(float(x1)), int(float(y1)), int(float(x2)), int(float(y2))])
            bbs.append(box)
        bbs = np.array(bbs)
        return bbs

    def _root_tag(self, fname):
        tree = parse(fname)
        root = tree.getroot()
        return root

    def _tree(self, fname):
        tree = parse(fname)
        return tree
    pass


class Annotation(object):
    def __init__(self, filename):
        self.fname = filename
        self.labels = []
        self.coded_labels = []
        self.boxes = None
        pass

    def add_object(self, x1, y1, x2, y2, name, code):
        self.labels.append(name)
        self.coded_labels.append(code)

        if self.boxes is None:
            self.boxes = np.array([x1, y1, x2, y2]).reshape(-1, 4)
        else:
            box = np.array([x1, y1, x2, y2]).reshape(-1, 4)
            self.boxes = np.concatenate([self.boxes, box])
        pass

    pass

def parse_annotation(ann_fname, img_dir, labels_name=[]):
    parser = PascalVocXmlParser()
    fname = parser.get_fname(ann_fname)

    annotation = Annotation(os.path.join(img_dir, fname))

    labels = parser.get_labels(ann_fname)
    boxes = parser.get_boxes(ann_fname)

    for label, box in zip(labels, boxes):
        x1, y1, x2, y2 = box
        if label in labels_name:
            annotation.add_object(x1, y1, x2, y2, name=label, code=labels_name.index(label))
    return annotation.fname, annotation.boxes, annotation.coded_labels

def iou(ground_truth_boxes, anchor_tensor_boxes):
    #ground_truth_boxs.shape=(n_gt_boxes,4) ，ground_truth_box[xmin,ymin,xmax,ymax]
    #anchor_boxs.shape=(n_boxes,4),anchor_box=[center_x,center_y,w,h]
    #return.shape=(n_gt_boxes,n_boxes)
    iou_matrix = np.zeros((len(ground_truth_boxes),len(anchor_tensor_boxes)))
    for i, gt_box in enumerate(ground_truth_boxes):
        for j, anchor_box in enumerate(anchor_tensor_boxes):
            gt_x0, gt_y0, gt_x1, gt_y1 = gt_box

            # 用左上角坐标以及右下角坐标表示anchor_box
            center_x, center_y, anchor_w, anchor_h = anchor_box
            anchor_x0 = center_x - anchor_w / 2
            anchor_y0 = center_y - anchor_h / 2
            anchor_x1 = center_x + anchor_w / 2
            anchor_y1 = center_y + anchor_h / 2

            int_x0 = max(gt_x0, anchor_x0)
            int_y0 = max(gt_y0, anchor_y0)
            int_x1 = min(gt_x1, anchor_x1)
            int_y1 = min(gt_y1, anchor_y1)

            int_area = max((int_x1 - int_x0),0) * max((int_y1 - int_y0),0)

            b1_area = max((gt_x1 - gt_x0),0) * max((gt_y1 - gt_y0),0)
            b2_area = max((anchor_x1 - anchor_x0),0) * max((anchor_y1 - anchor_y0),0)

            # 分母加个1e-05，避免除数为 0
            iou_ = int_area / (b1_area + b2_area - int_area + 1e-05)
            iou_matrix[i][j] = iou_
            pass
        pass
    return iou_matrix
    pass
