from math import ceil
import time
import numpy as np
import cv2
import logging


def pre_process_batch_yolox(image_list, max_batch=1, img_size=(640, 640), swap=(2, 0, 1), un_read=False):
    """
    return: [[preprocessed images], [source images]]
    """
    exp_width, exp_height = img_size[0], img_size[1]
    group_num = ceil(len(image_list) / max_batch)
    for num in range(group_num):
        ST_time = time.time()
        output = [np.ones((3, exp_height, exp_width), dtype=np.float32) * 114 for _ in range(max_batch)]
        for index, img in enumerate(image_list[num * max_batch: (num * max_batch) + max_batch]):
            if un_read:
                img = cv2.imread(img, cv2.IMREAD_COLOR)
            # 创建一个(640, 640, 3)的数组
            padded_img = np.ones((exp_height, exp_width, 3), dtype=np.uint8) * 114
            # 计算图片实际大小和预期大小插值
            r = min(exp_height / img.shape[0], exp_width / img.shape[1])
            # resize图片
            resized_img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)),
                                     interpolation=cv2.INTER_LINEAR).astype(np.uint8)
            # 填充resized图片到padded_img
            padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
            # 转换成(3, 640, 640的数组)
            padded_img = padded_img.transpose(swap)
            output[index] = padded_img
        output = np.array(output)
        # 转换数组位置到内存连续， 加速调用
        logging.debug("preprocess batch: {}s".format(time.time() - ST_time))
        yield [np.ascontiguousarray(output, dtype=np.float32), image_list[num * max_batch: (num * max_batch) + max_batch]]


def pre_process_batch_yolov7(image_list, max_batch=1, img_size=(640, 640), swap=(2, 0, 1), un_read=False):
    """
    return: [[preprocessed images], [source images]]
    """
    exp_width, exp_height = img_size[0], img_size[1]
    group_num = ceil(len(image_list) / max_batch)
    for num in range(group_num):
        ST_time = time.time()
        output = [np.ones((3, exp_height, exp_width), dtype=np.float32) * 114 for _ in range(max_batch)]
        for index, img in enumerate(image_list[num * max_batch: (num * max_batch) + max_batch]):
            if un_read:
                img = cv2.imread(img, cv2.IMREAD_COLOR)
            # 创建一个(640, 640, 3)的数组
            padded_img = np.full((exp_height, exp_width, 3), fill_value=128, dtype=np.uint8) * 114
            # 计算图片实际大小和预期大小插值
            r = min(exp_height / img.shape[0], exp_width / img.shape[1])
            # resize图片
            resized_img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)),
                                     interpolation=cv2.INTER_LINEAR).astype(np.uint8)
            # 填充resized图片到padded_img
            padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
            # 转换数据类型
            padded_img = padded_img.astype(np.float32)
            # 归一化
            padded_img /= 255.0
            # 转换成(3, 640, 640的数组) 即CHW格式
            padded_img = padded_img.transpose(swap)
            output[index] = padded_img
        # CHW 到 NCHW 格式
        # for index in range(len(output)):
        #     output[index] = np.expand_dims(output[index], axis=0)
        output = [np.expand_dims(out, axis=0) for out in output]

        output = np.array(output, dtype=np.float32)
        # 转换数组位置到内存连续， 加速调用
        logging.debug("preprocess batch: {}s".format(time.time() - ST_time))
        yield [np.ascontiguousarray(output, dtype=np.float32), image_list[num * max_batch: (num * max_batch) + max_batch]]
