#-- coding: utf-8 --
#在跟了batchnorm层的卷积层设置偏置是多此一举；
import tensorflow as tf

def match_1(matrix,ground_truth_boxes):
    # 匹配原则1：将与ground_truth_box的iou最大的anchor_tensor用来当作该ground_truth_box的正例
    idxs_x_matrix = []
    for i, item in enumerate(matrix):
        idxs_x = np.argsort(-item)
        idxs_x_matrix.append(list(idxs_x))
        pass
    idxs_x_matrix = np.array(idxs_x_matrix)
    while (len(idxs_x_matrix) > 0):
        idxs_a = np.argsort(-idxs_x_matrix[:, 0])
        idxs_x_matrix = idxs_x_matrix[idxs_a]
        #coded_labels = coded_labels[idxs_a]
        ground_truth_boxes = ground_truth_boxes[idxs_a]
        print(ground_truth_boxes[0])
        print(idxs_x_matrix[0, 0])
        #y_encode_template[idxs_x_matrix[0, 0], 0] = 0
        #y_encode_template[idxs_x_matrix[0, 0], coded_labels[0]] = 1

        # 坐标转换
        # x1, y1, x2, y2 = ground_truth_boxes[0]
        # center_x = (x1 + x2) / 2
        # center_y = (y1 + y2) / 2
        # gt_w = x2 - x1
        # gt_h = y2 - y1
        # y_encode_template[idxs_x_matrix[0, 0], -12:-8] = [center_x, center_y, gt_w, gt_h]

        # 从idxs_x_matrix中删除idxs_x_matrix[0]
        t = idxs_x_matrix[0, 0]
        idxs_x_matrix = idxs_x_matrix[1:]
        #coded_labels = coded_labels[1:]
        ground_truth_boxes = ground_truth_boxes[1:]

        # 从idxs_x_matrix中删除temp
        idxs_x_matrix_ = []
        for item in idxs_x_matrix:
            item = [x for x in item if x != t]
            idxs_x_matrix_.append(item)
            pass

        idxs_x_matrix = np.array(idxs_x_matrix_)
    pass

def match_one(matrix):
    weight_matrix = np.copy(matrix)  # We'll modify this array.
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
        print(ground_truth_index)
        anchor_index = anchor_indices[ground_truth_index]
        print(anchor_index)
        matches[ground_truth_index] = anchor_index  # Set the match.

        # Set the row of the matched ground truth box and the column of the matched
        # anchor box to all zeros. This ensures that those boxes will not be matched again,
        # because they will never be the best matches for any other boxes.
        weight_matrix[ground_truth_index] = 0
        weight_matrix[:, anchor_index] = 0
        pass
    pass

if __name__ == "__main__":
    import numpy as np
    matrix = np.array([[3, 2, 4, 1, 5], [5, 3, 2, 1, 4], [5, 1, 3, 2, 4]])
    match_one(matrix)
    print("=============================================================")
    match_1(matrix,np.array([[0,1,2],[3,4,5],[6,7,8]]))

    pass

