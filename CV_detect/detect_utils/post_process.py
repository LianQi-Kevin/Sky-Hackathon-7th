import numpy as np


def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        index = np.where(ovr <= nms_thr)[0]
        order = order[index + 1]
    return keep


def multiclass_nms(boxes, scores, nms_thr=0.45, score_thr=0.3):
    """Multiclass NMS implemented in Numpy"""
    final_detects = []
    num_classes = scores.shape[1]
    for cls_index in range(num_classes):
        # 抽取类别维度, 单类
        cls_scores = scores[:, cls_index]
        # 判断单类得分超过conf
        valid_score_mask = cls_scores > score_thr
        # 判断valid_score_mask中True的数量
        if valid_score_mask.sum() == 0:
            continue
        else:
            # 取出对应的scores和boxes
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            # 单类NMS
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_indexes = np.ones((len(keep), 1)) * cls_index
                detects = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_indexes], 1
                )
                final_detects.append(detects)
    if len(final_detects) == 0:
        return None
    return np.concatenate(final_detects, 0)


def post_process_batch(host_outputs, batch_size=1, conf=0.3, nms=0.45, num_class=80, result_path=None):
    """
    :return [[x1, y1, x2, y2, scores, cls_name], [x1, y1, x2, y2, scores, cls_name], ···]
    """
    if result_path is not None:
        pass

    # xywh2xyxy (4ms)
    team_num = num_class + 5
    prediction = host_outputs.reshape(batch_size, int(host_outputs.shape[0] / team_num / batch_size), team_num)
    box_corner = np.ones_like(prediction)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    # get detections
    output = [None for _ in range(len(prediction))]
    for index, image_pred in enumerate(prediction):
        boxes = image_pred[:, :4]
        # 获得每个clss的score
        scores = image_pred[:, 4:5] * image_pred[:, 5:]
        output[index] = multiclass_nms(boxes, scores, nms_thr=nms, score_thr=conf)
    return output


if __name__ == '__main__':
    pass