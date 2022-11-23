import copy
import math
import os
import threading
import time
import uuid

import cv2
import numpy as np
from torch import cat as torch_cat
from torch import max as torch_max
from torch.tensor import Tensor
from torchvision.ops import batched_nms
from tqdm import tqdm
from yolox.exp import get_exp

from utils.trtpy_detect import get_image_list, make_args, YOLOX_TRT_Detection


# 因需要继承全局变量 故于类外单独定义
def postprocess_async(img_group, idx, batch_size=1, conf=0.3, nms=0.45, num_classes=4):
    # 调用全局变量
    global detection, global_outputs, post_cpy_finish
    print("Start post process {} ".format(idx))

    # copy mem
    post_cpy_finish = False
    detection.stream.synchronize()
    # print(idx, detection.host_outputs[0].shape)
    team_num = num_classes + 5
    host_outputs = np.copy(detection.host_outputs[0], order="C")
    post_cpy_finish = True

    # xywh2xyxy
    prediction = host_outputs.reshape((batch_size, int(host_outputs.shape[0] / team_num / batch_size), team_num))
    box_corner = np.zeros(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]
    prediction = Tensor(prediction)

    # get detections
    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):
        class_conf, class_pred = torch_max(image_pred[:, 5: team_num], dim=1, keepdim=True)
        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf).squeeze()
        detections = torch_cat([image_pred[:, :4],
                                image_pred[:, 4].reshape(int(host_outputs.shape[0] / team_num / batch_size),
                                                         1) * class_conf, class_pred.float()], dim=1)
        detections = detections[conf_mask]

        # iou nms
        nms_out_index = batched_nms(
            boxes=detections[:, :4],
            scores=detections[:, 4],
            idxs=detections[:, 5],
            iou_threshold=nms)

        output[i] = detections[nms_out_index]
    global_outputs[idx] = [[out.tolist() for out in output], img_group]


def main_dir_async_batch(args):
    # 调用全局变量
    global global_outputs, img_path_group, detection, result_path, post_cpy_finish

    # 验证图片文件夹和模型是否存在
    assert os.path.exists(args.images_dir), "{} not exists".format(args.images_dir)
    assert os.path.exists(args.trt_path), "{} not exists".format(args.trt_path)

    # get img list
    if un_pre_read_img:
        image_list = get_image_list(args.images_dir)
    else:
        image_list = [cv2.imread(img, cv2.IMREAD_COLOR) for img in tqdm(get_image_list(args.images_dir))]
    print("{} images in {}".format(len(image_list), args.images_dir))

    thread = []
    old_image_group = []

    ST_time = time.time()
    for index, preprocess_return in enumerate(
            detection.pre_process_batch(image_list, args.batch_size, un_read=un_pre_read_img)):
        # preprocess
        img_group = preprocess_return[0]
        source_images = preprocess_return[1]
        print(img_group.shape)

        detection.stream.synchronize()

        # postprocess [0:-2]
        if index != 0:
            while index != 1:
                if post_cpy_finish:
                    break
                time.sleep(0.01)
            thread.append(threading.Thread(
                target=postprocess_async,
                args=(old_image_group, index - 1, args.batch_size, 0.3, 0.45, len(args.cls_list))).start())
            old_image_group = copy.copy(source_images)

        # detect
        detection.detect(img_group)

        # postprocess [0]
        if index == 0:
            old_image_group = copy.copy(source_images)

        # postprocess [-1]
        if index == len(global_outputs) - 1:
            detection.stream.synchronize()
            postprocess_async(source_images, index, args.batch_size, 0.3, 0.45, len(args.cls_list))

    # wait postprocess all done
    while None in global_outputs:
        time.sleep(0.1)

    print(time.time() - ST_time)

    for num, output in enumerate(global_outputs):
        out = output[0]
        img_group = output[1]

        if un_pre_read_img:
            img_path_group = image_list[num * args.batch_size: (num * args.batch_size) + args.batch_size]
        for a in range(len(img_group)):
            if un_pre_read_img:
                vis_img = detection.visual(output=out[a], img=cv2.imread(img_path_group[a], cv2.IMREAD_COLOR),
                                           cls_conf=0.35)
            else:
                vis_img = detection.visual(output=out[a], img=img_group[a], cls_conf=0.35)
            cv2.imwrite(os.path.join(result_path, "{}_{}.jpg".format(num, uuid.uuid4())), vis_img)


if __name__ == '__main__':
    # args
    args = make_args()

    # 主定义参数
    args.exp_file = "./selfEXP.py"
    args.trt_path = "models/yolox-s/yolox-s-batch8_upsample_dynamic.fp16.trt"
    args.fp16 = True
    args.result_path = "./result_images"
    args.images_dir = "./face_mask_images"
    args.batch_size = 8
    args.cls_list = ["banana", "CARDBOARD", "bottle"]

    # ------------------- 因需要继承全局变量 故定义于此 ---------------------------------
    global_outputs = [None for _ in range(math.ceil(len(os.listdir(args.images_dir)) / args.batch_size))]
    post_cpy_finish = False
    un_pre_read_img = False
    # 加载检测器
    MyExp = get_exp(exp_file=args.exp_file)
    detection = YOLOX_TRT_Detection(
        engine_file_path=args.trt_path,
        exp=MyExp,
        cls_list=args.cls_list,
        batch_size=args.batch_size,
        fp16=args.fp16)

    result_path = os.path.join(args.result_path,
                               "{}_{}".format(MyExp.exp_name, time.strftime("%Y%m%d-%H%M%S", time.localtime())))
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    # ------------------- 因需要继承全局变量 故定义于此 ---------------------------------

    main_dir_async_batch(args)
    # main_dir_serial_batch(args)
