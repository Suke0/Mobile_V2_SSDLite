#-- coding: utf-8 --
import cv2
from PIL import ImageDraw,Image
import numpy as np

def get_anchor_sizes(min_scale=0.1, max_scale=0.9,max_n_anchors=6,net_size=300):
    scales = np.linspace(min_scale, max_scale, max_n_anchors + 1)
    sizes = scales * net_size #[ 30.  70. 110. 150. 190. 230. 270.]
    anchor_sizes = []
    for i in range(0,len(sizes)-1):
        anchor_sizes.append([round(sizes[i]),round(sizes[i+1])])
        pass
    return anchor_sizes
    pass

def get_anchors_wh(anchor_sizes,anchor_ratios):
    anchors = []
    for anchor_size, anchor_ratio in zip(anchor_sizes, anchor_ratios):
        feat_anchors = []
        feat_anchors.append([anchor_size[0], anchor_size[0]])
        for val in anchor_ratio:
            feat_anchors.append([round(anchor_size[0] * np.sqrt(val)), round(anchor_size[0] / np.sqrt(val))])
            pass
        feat_anchors.append([round(np.sqrt(anchor_size[0] * anchor_size[1])), round(np.sqrt(anchor_size[0] * anchor_size[1]))])
        anchors.append(feat_anchors)
        pass
    return np.array(anchors)
    pass

#计算anchor_for_layer,即(cx,cy,pw,ph),anchor_for_layer.shape(batch_size,img_w,img_h,n_anchors,4+4)
def get_encode_anchor_for_layer(shape,anchors_wh,variances=[0.1,0.1,0.2,0.2],net_size=300):
    #shape=(batch_size,img_w,img_h,n_anchors,4)
    grid_x = range(0,shape[1])
    grid_y = range(0,shape[2])
    offset_x,offset_y = np.meshgrid(grid_x,grid_y)
    offset_x = np.reshape(offset_x,(-1,1))
    offset_y = np.reshape(offset_y,(-1,1))
    offset_xy = np.concatenate([offset_x,offset_y],-1)
    offset_xy = np.around(offset_xy * net_size / shape[1])

    offset_xy = np.expand_dims(offset_xy,1)
    offset_xy = np.tile(offset_xy,(1,shape[3],1))

    offset_xy = np.reshape(offset_xy, (shape[1], shape[2],shape[3],2))

    anchors_wh = np.reshape(anchors_wh,(1, 1,shape[3],2))
    anchors_wh = np.tile(anchors_wh,(shape[1], shape[2],1,1))

    anchors_xywh = np.concatenate([offset_xy,anchors_wh],-1)
    variances = np.reshape(variances,(1,1,1,4))
    variances = np.tile(variances,(shape[1],shape[2],shape[3],1))
    encoded_anchors = np.concatenate([anchors_xywh,variances],-1) #(batch_size,img_w,img_h,n_anchors,8)
    encoded_anchors = np.reshape(encoded_anchors,(1,np.prod(shape[1:4]),4+4))
    encoded_anchors = np.tile(encoded_anchors,(shape[0],1,1))
    return encoded_anchors #(batch_size,img_w,img_h,n_anchors,8)
    pass


def match_bipartite_greedy(weight_matrix):
    weight_matrix = np.copy(weight_matrix)  # We'll modify this array.
    num_ground_truth_boxes = weight_matrix.shape[0]
    all_gt_indices = list(range(num_ground_truth_boxes))  # Only relevant for fancy-indexing below.

    matches = np.zeros(num_ground_truth_boxes, dtype=np.int)

    # In each iteration of the loop below, exactly one ground truth box
    # will be matched to one anchor box.
    for _ in range(num_ground_truth_boxes):
        # Find the maximal anchor-ground truth pair in two steps: First, reduce
        # over the anchor boxes and then reduce over the ground truth boxes.
        anchor_indices = np.argmax(weight_matrix, axis=1)  # Reduce along the anchor box axis.
        overlaps = weight_matrix[all_gt_indices, anchor_indices]
        ground_truth_index = np.argmax(overlaps)  # Reduce along the ground truth box axis.
        anchor_index = anchor_indices[ground_truth_index]
        matches[ground_truth_index] = anchor_index  # Set the match.

        # Set the row of the matched ground truth box and the column of the matched
        # anchor box to all zeros. This ensures that those boxes will not be matched again,
        # because they will never be the best matches for any other boxes.
        weight_matrix[ground_truth_index] = 0
        weight_matrix[:, anchor_index] = 0

    return matches
    pass


def match_multi(weight_matrix, threshold):
    num_anchor_boxes = weight_matrix.shape[1]
    all_anchor_indices = list(range(num_anchor_boxes))  # Only relevant for fancy-indexing below.

    # Find the best ground truth match for every anchor box.
    ground_truth_indices = np.argmax(weight_matrix, axis=0)  # Array of shape (weight_matrix.shape[1],)
    overlaps = weight_matrix[ground_truth_indices, all_anchor_indices]  # Array of shape (weight_matrix.shape[1],)

    # Filter out the matches with a weight below the threshold.
    anchor_indices_thresh_met = np.nonzero(overlaps >= threshold)[0]
    gt_indices_thresh_met = ground_truth_indices[anchor_indices_thresh_met]

    return gt_indices_thresh_met, anchor_indices_thresh_met
    pass

def resize_image(image,boxes,img_w,img_h):#对原始图片进行缩放，并且重新计算True box的位置坐标
    h,w,_ = image.shape #原图片的真实高宽
    # resize the image to standard size
    image = cv2.resize(image,(img_h,img_w))#原图片缩放后的高宽
    image = image[:,:,::-1] #cv2把图片读取后是把图片读成BGR形式的,img[：，：，：：-1]的作用就是实现BGR到RGB通道的转换
    # fix object's position and size
    new_boxes = []
    for box in boxes:
        x1,y1,x2,y2 = box
        x1 = int(x1 * float(img_w) / w)
        x1 = max(min(x1,img_w),0)
        x2 = int(x2 * float(img_w) / w)
        x2 = max(min(x2,img_w),0)

        y1 = int(y1 * float(img_h) / h)
        y1 = max(min(y1, img_h), 0)
        y2 = int(y2 * float(img_h) / h)
        y2 = max(min(y2, img_h), 0)
        new_boxes.append([x1,y1,x2,y2])
    return image, np.array(new_boxes)
    pass


def two_boxes_iou(box1, box2):
    b1_x0, b1_y0, b1_x1, b1_y1 = box1
    b2_x0, b2_y0, b2_x1, b2_y1 = box2

    int_x0 = max(b1_x0, b2_x0)

    int_y0 = max(b1_y0, b2_y0)
    int_x1 = min(b1_x1, b2_x1)
    int_y1 = min(b1_y1, b2_y1)

    int_area = max((int_x1 - int_x0),0) * max((int_y1 - int_y0),0)

    b1_area = max((b1_x1 - b1_x0),0) * max((b1_y1 - b1_y0),0)
    b2_area = max((b2_x1 - b2_x0),0) * max((b2_y1 - b2_y0),0)

    # 分母加个1e-05，避免除数为 0
    iou = int_area / (b1_area + b2_area - int_area + 1e-05)
    return iou



def encode_box(y_encode_template):
    #1、将与ground_truth_box的IOU最大的anchor对应的预测框，归为正例
    #2、将与ground_truth_box的IOU大于threshhold的anchor对应的预测框，归为正例
    #3、若同一个anchor对应多个ground_truth_box，则将其归为IOU最大的ground_truth_box的正例
    #4、将与ground_truth_box的IOU位于[neg_iou_threshold，pos_iou_threshold]之间的anchor对应的预测框抛弃

    # (tx,ty,tw,th, cx,cy,pw,ph, v0,v1,v2,v3)
    # tx = (bx-cx)/(pw*v0)
    # ty = (by-cy)/(ph*v1)
    # tw = 1/v2 * log(bw/pw)
    # th = 1/v3 * log(bh/ph)
    idxs = y_encode_template[:, 0] == 0
    # tx
    y_encode_template[idxs, -12] -= y_encode_template[idxs, -8]
    y_encode_template[idxs, -12] /= y_encode_template[idxs, -6] * y_encode_template[idxs, -4]

    # ty
    y_encode_template[idxs, -11] -= y_encode_template[idxs, -7]
    y_encode_template[idxs, -11] /= y_encode_template[idxs, -5] * y_encode_template[idxs, -3]

    # tw
    y_encode_template[idxs, -10] = np.log(y_encode_template[idxs, -10] / y_encode_template[idxs, -6] + 1e-5)
    y_encode_template[idxs, -10] /= y_encode_template[idxs, -2]

    # th
    y_encode_template[idxs, -9] = np.log(y_encode_template[idxs, -9] / y_encode_template[idxs, -5] + 1e-5)
    y_encode_template[idxs, -9] /= y_encode_template[idxs, -1]
    # print(len(y_encode_template[mask]))
    pass


def decode_box(y_encode_template):
    # (tx,ty,tw,th, cx,cy,pw,ph, v0,v1,v2,v3)
    # bx = tx*v0*pw+cx
    # by = ty*v1*ph+cy
    # bw = pw*exp(tw*v2)
    # bh = ph*exp(th*v3)

    # tx, ty
    y_encode_template[:, [-12,-11]] *= y_encode_template[:, [-6,-5]] * y_encode_template[:, [-4,-3]]
    y_encode_template[:, [-12,-11]] += y_encode_template[:, [-8,-7]]

    # tw, th
    y_encode_template[:, [-10,-9]] = np.exp(y_encode_template[:, [-2,-1]] * y_encode_template[:, [-10,-9]])
    y_encode_template[:, [-10,-9]] *= y_encode_template[:, [-6,-5]]
    pass



#定义函数：将中心点、高、宽坐标 转化为[x0, y0, x1, y1]坐标形式
def coordinate_translate(detections_,net_size=300):
    label, prod, center_x, center_y, width, height= np.split(detections_, 6, axis=-1)
    w2 = width / 2
    h2 = height / 2
    x0 = np.maximum(np.minimum(center_x - w2,net_size),0)
    y0 = np.maximum(np.minimum(center_y - h2,net_size),0)
    x1 = np.maximum(np.minimum(center_x + w2,net_size),0)
    y1 = np.maximum(np.minimum(center_y + h2,net_size),0)

    boxes = np.concatenate([x0, y0, x1, y1], axis=-1)
    detections = np.concatenate([label, prod,boxes], axis=-1)
    return detections

#使用NMS方法，对结果去重
def non_max_suppression(predictions_boxes, confidence_threshold=0.5, iou_threshold=0.4):
    mask = predictions_boxes[:, 0:-12] > confidence_threshold
    predictions_boxes[:, 0:-12] = predictions_boxes[:, 0:-12] * mask
    idxs = np.nonzero(predictions_boxes[:, 0:-12])
    idxs = list(set(idxs[0]))
    predictions = predictions_boxes[idxs]

    labels = np.argmax(predictions[:, 0:-12], -1)
    predictions = predictions[labels > 0]
    labels = labels[labels > 0]
    decode_box(predictions)
    predictions_ = np.copy(predictions[:, -14:-8])
    predictions_[:, 0] = labels
    predictions_[:, 1] = [predictions[i,x] for i,x in enumerate(labels)]

    predictions_ = coordinate_translate(predictions_)
    labels_ = list(set(labels))
    result = []
    print(f'正例样本数：{len(predictions_)}')
    for label in labels_:
        idxs = predictions_[:, 0] == label
        label_pred_boxes = predictions_[idxs]
        while len(label_pred_boxes) > 0:
            idxs = np.argsort(-label_pred_boxes[:, 1])  # 降序排序
            label_max_box = label_pred_boxes[idxs[0]]
            label_pred_boxes = label_pred_boxes[idxs[1:]]
            result.append(label_max_box)
            box1 = label_max_box[2:6]

            for i, box2 in enumerate(label_pred_boxes[:, 2:6]):
                iou = two_boxes_iou(box1, box2)
                if iou > iou_threshold:
                    label_pred_boxes[i,0] = 0
                    pass
            label_pred_boxes = label_pred_boxes[label_pred_boxes[:,0] > 0]
            if len(label_pred_boxes) == 1:
                label_pred_boxes = np.reshape(label_pred_boxes,(1,6))
                pass
            pass
    return np.array(result) # (n_boxes, 1+4)


def convert_to_original_size(box, size, original_size):
    ratio = original_size / size
    box = box.reshape(2, 2) * ratio
    return list(box.reshape(-1))


# 将级别结果显示在图片上
def draw_boxes(i, boxes, img_file, cls_names, detection_size):
    img = Image.open(img_file)
    draw = ImageDraw.Draw(img)

    for box in boxes:
        box[2:] = convert_to_original_size(np.copy(box[2:]), np.array(detection_size), np.array(img.size))
        draw.rectangle(list(box[2:]), outline='red')
        draw.text(list(box[2:4]),'{} {:.2f}%'.format(cls_names[int(box[0])], box[1] * 100), fill='red')
        print('{} {:.2f}%'.format(cls_names[int(box[0])], box[1] * 100), list(box[2:]))
    img.save(f"output_img{i}.jpg")
    img.show()
    pass


def match_max_iou(iou_matrix,y_encode_template,ground_truth_boxes,coded_labels):
    # 匹配原则1：将与ground_truth_box的iou最大的anchor_tensor用来当作该ground_truth_box的正例
    idxs_x_matrix = []
    for i, item in enumerate(iou_matrix):
        idxs_x = np.argsort(-item)
        idxs_x_matrix.append(list(idxs_x))
        pass
    idxs_x_matrix = np.array(idxs_x_matrix)
    while (len(idxs_x_matrix) > 0):
        idxs_a = np.argsort(-idxs_x_matrix[:, 0])
        idxs_x_matrix = idxs_x_matrix[idxs_a]
        coded_labels = coded_labels[idxs_a]
        ground_truth_boxes = ground_truth_boxes[idxs_a]
        y_encode_template[idxs_x_matrix[0, 0], 0] = 0
        y_encode_template[idxs_x_matrix[0, 0], coded_labels[0]] = 1

        # 坐标转换
        x1, y1, x2, y2 = ground_truth_boxes[0]
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        gt_w = x2 - x1
        gt_h = y2 - y1
        y_encode_template[idxs_x_matrix[0, 0], -12:-8] = [center_x, center_y, gt_w, gt_h]

        # 从idxs_x_matrix中删除idxs_x_matrix[0]
        t = idxs_x_matrix[0, 0]
        idxs_x_matrix = idxs_x_matrix[1:]
        coded_labels = coded_labels[1:]
        ground_truth_boxes = ground_truth_boxes[1:]

        # 从idxs_x_matrix中删除temp
        idxs_x_matrix_ = []
        for item in idxs_x_matrix:
            item = [x for x in item if x != t]
            idxs_x_matrix_.append(item)
            pass

        idxs_x_matrix = np.array(idxs_x_matrix_)
    pass


def match_many(iou_matrix,y_encode_template,ground_truth_boxes,coded_labels,pos_iou_threshold=0.5):
    # 匹配原则2：将与ground_truth_box的iou大于阈值的anchor_tensor用来当作该ground_truth_box的正例
    mask = iou_matrix >= pos_iou_threshold
    idxs_tuple = np.nonzero(mask)
    idxs = list(set(idxs_tuple[1])) # [2116, 2117, 2121, 556, 558, 559, 2260, 2261]
    idxs_ = []
    for id in idxs:
        if y_encode_template[id,0] == 1:
            idxs_.append(id)
            pass
        pass

    iou_matrix_ = iou_matrix[:, idxs_]

    idxs_max = np.argmax(iou_matrix_, 0)
    y_encode_template[idxs_,coded_labels[idxs_max]] = 1
    y_encode_template[idxs_, 0] = 0

    for i, id in enumerate(idxs_):
        x1, y1, x2, y2 = ground_truth_boxes[idxs_max[i]]
        print(ground_truth_boxes[idxs_max[i]])
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        gt_w = x2 - x1
        gt_h = y2 - y1
        y_encode_template[id, -12:-8] = [center_x, center_y, gt_w, gt_h]
        pass
    pass