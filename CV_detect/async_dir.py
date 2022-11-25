import logging
import os
import threading
import time
from copy import deepcopy
from math import ceil

import cv2

from detect_utils.pre_process import pre_process_batch_yolov7
from detect_utils.trtpy_detect import TRT_Detection
from detect_utils.utils import log_set, get_image_list, make_args, to_mAP

# Global var
preprocessed_img_group = list()
postprocess_result_group = list()
visual_result_group = list()


def set_global_var(img_num: int, batch_size: int):
    global preprocessed_img_group, postprocess_result_group, visual_result_group
    batch_group_num = ceil(img_num / batch_size)
    preprocessed_img_group = [None for _ in range(batch_group_num)]
    postprocess_result_group = [None for _ in range(batch_group_num)]
    visual_result_group = [None for _ in range(batch_group_num)]


# 封装预处理函数到多线程样式
def multi_thread_preprocess(preprocess_func, img_list: list, batch_size: int, un_read=False):
    global preprocessed_img_group
    for index, preprocess_group in enumerate(preprocess_func(img_list, batch_size, (640, 640), un_read=un_read)):
        preprocessed_img_group[index] = preprocess_group
        logging.info("Finish preprocess image group {}, Img No.{} ~ No.{}".format(index, batch_size * index,
                                                                                  batch_size * index + batch_size - 1))


# 封装检测函数到多线程样式
def multi_thread_detection():
    global preprocessed_img_group, postprocess_result_group, detection
    for index in range(len(postprocess_result_group)):
        while preprocessed_img_group[index] is None:
            time.sleep(0.01)
        logging.debug("Start detect group {}".format(index))
        output = detection.detect(preprocessed_img_group[index][0])
        postprocess_result_group[index] = deepcopy(output)
        logging.info("Finish detect group {}".format(index))
    detection.destroy()


# 封装后处理函数到多线程样式
def multi_thread_postprocess(batch_size=1, conf=0.3, nms=0.45):
    global postprocess_result_group, visual_result_group, detection
    for index in range(len(postprocess_result_group)):
        while postprocess_result_group[index] is None:
            time.sleep(0.01)
        logging.debug("Start postprocess group {}".format(index))
        output = detection.post_process_batch(host_outputs=postprocess_result_group[index],
                                              batch_size=batch_size, conf=conf, nms=nms)
        output = [out.tolist() for out in output]
        visual_result_group[index] = deepcopy(output)
        logging.info("Finish postprocess group {}".format(index))


# 封装画图函数到多线程样式
def multi_thread_visual(img_path_list, batch_size=1, img_result_path="./result_images", mAP_result_path="mAP/input/detection-results"):
    global preprocessed_img_group, visual_result_group, detection, args
    for index in range(len(visual_result_group)):
        while visual_result_group[index] is None:
            time.sleep(0.01)
        logging.debug("Start visual group {}".format(index))
        images_path = img_path_list[index * batch_size: index * batch_size + batch_size]
        for single_out, source_img, img_path in zip(visual_result_group[index], preprocessed_img_group[index][1],
                                                    images_path):
            # write img
            vis_img = deepcopy(detection.visual(single_out, source_img, cls_conf=0.35))
            write_path = os.path.join(img_result_path, os.path.basename(img_path))
            cv2.imwrite(write_path, vis_img)

            # write mAP
            txt_path = os.path.join(mAP_result_path, "{}.txt".format(os.path.splitext(os.path.basename(img_path))[0]))
            with open(txt_path, "w") as mAP_f:
                if single_out:
                    bandboxes, scores, classes = detection.remapping_result(single_out, source_img)
                    mAP_list = to_mAP(bandboxes, scores, classes, detection.cls_list)
                    for text in mAP_list:
                        mAP_f.write("{} {} {} {} {} {}".format(text[0], text[1], text[2], text[3], text[4], text[5]))
                        if text != mAP_list[-1]:
                            mAP_f.write("\n")
                else:
                    mAP_f.write("")
                mAP_f.close()

        logging.info("Finish visual group {}".format(index))


if __name__ == '__main__':
    # logging
    log_set(screen_show=logging.INFO)

    # args
    args = make_args()

    # 主定义参数
    args.trt_path = "models/yolov7_rep/yolov7_rep_grid_simplify.fp16.trt"
    args.result_path = "./result_images"
    args.images_dir = "./mAP/input/images-optional"
    args.batch_size = 8
    args.cls_list = ['CARDBOARD', 'banane', 'bottle']
    args.result_basename = "test_detect"
    args.conf = 0.1
    args.nms = 0.45

    # load model
    detection = TRT_Detection(
        engine_file_path=args.trt_path,
        cls_list=args.cls_list,
        batch_size=args.batch_size,
        exp_size=(640, 640)
    )

    # set global var
    args.img_path_list = get_image_list(args.images_dir)
    args.img_list = [cv2.imread(path) for path in args.img_path_list]
    set_global_var(img_num=len(args.img_list), batch_size=args.batch_size)

    # threads pool
    threads = [None for _ in range(4)]

    # preprocess thread
    threads[0] = (threading.Thread(target=multi_thread_preprocess,
                                   args=(pre_process_batch_yolov7, args.img_list, args.batch_size, False)).start())

    # detection thread
    threads[1] = (threading.Thread(target=multi_thread_detection).start())

    # postprocess thread
    threads[2] = (
        threading.Thread(target=multi_thread_postprocess, args=(args.batch_size, args.conf, args.nms)).start())

    threads[3] = (
        threading.Thread(target=multi_thread_visual, args=(args.img_path_list, 8, args.result_path)).start())
