import os
import shutil
import time

import cv2
from loguru import logger
from yolox.exp import get_exp
from tqdm import tqdm
import argparse

from utils.trtpy_for_YOLOX_async import YOLOX_TRT_Detection, get_image_list


@logger.catch
def main_dir_serial_batch(args):
    # 如果待检测文件夹存在则创建文件列表
    assert os.path.exists(args.images_dir), "{} not exists".format(args.images_dir)
    assert os.path.exists(args.trt_path), "{} not exists".format(args.trt_path)
    image_list = get_image_list(args.images_dir)
    # image_list = [cv2.imread(img, cv2.IMREAD_COLOR) for img in get_image_list(args.images_dir)]
    print("{} images in {}".format(len(image_list), args.images_dir))

    # 重建txt_label输出位置
    if os.path.exists(args.mAP_label_result):
        shutil.rmtree(args.mAP_label_result)
    os.makedirs(args.mAP_label_result)

    # 加载检测器
    MyExp = get_exp(exp_file=args.exp_file)
    detection = YOLOX_TRT_Detection(
        engine_file_path=args.trt_path,
        exp=MyExp,
        cls_list=args.cls_list,
        batch_size=args.batch_size,
        fp16=args.fp16)

    # draw output
    result_path = os.path.join(args.result_path,
                               "{}_{}".format(MyExp.exp_name, time.strftime("%Y%m%d-%H%M%S", time.localtime())))
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    outputs = []
    async_time = time.time()
    for index, preprocess_return in enumerate(detection.pre_process_batch(image_list, args.batch_size, un_read=True)):
        img_group = preprocess_return[0]
        source_images = preprocess_return[1]
        img_path_list = image_list[index * args.batch_size: (index * args.batch_size) + args.batch_size]

        # detect
        ST_time = time.time()
        print("start detect: {} - {}".format(index, ST_time))
        output = detection.detect(img_group)
        detection.stream.synchronize()
        FINDET_time = time.time()
        print("finish detect {} - {}".format(index, FINDET_time - ST_time))
        # postprocess return [x1, y1, x2, y2, scores, cls_num]
        output = detection.post_process_batch(host_outputs=output, batch_size=args.batch_size)
        print("post process: {} - {}".format(index, time.time() - FINDET_time))
        outputs.append([[out.tolist() for out in output], source_images, img_path_list])
        print([out.tolist() for out in output])
        # pass
        print("once time: {}".format(time.time() - ST_time))

    for num, output in tqdm(enumerate(outputs), total=len(outputs), ):
        out = output[0]
        img_group = [cv2.imread(img) for img in output[1]]
        img_path = output[2]

        for index, file_path in enumerate(img_path):
            # print(file_path)
            with open(os.path.join(args.mAP_label_result,
                                   "{}.txt".format(os.path.splitext(os.path.basename(img_path[index]))[0])), "w") as gt_f:
                bandboxes, scores, classes = detection.remapping_result(out[index], img_group[index])
                for bandbox in zip(classes.tolist(), scores.tolist(), bandboxes.tolist()):
                    result = "{} {} {} {} {} {}".format(args.cls_list[int(bandbox[0])], bandbox[1], bandbox[2][0],
                                                        bandbox[2][1], bandbox[2][2], bandbox[2][3])
                    gt_f.write(result)
                    gt_f.write("\n")
                    # print(result)
            gt_f.close()

            vis_img = detection.visual(output=out[index], img=img_group[index], cls_conf=0.1)
            cv2.imwrite(os.path.join(result_path, os.path.split(img_path[index])[1]), vis_img)

    print("async time: {}".format(time.time() - async_time))
    print("output shape: {}".format(len(outputs)))


def make_args():
    parser = argparse.ArgumentParser("YOLOX TRT DETECT!")
    parser.add_argument("-e", "--trt_path", default=None, type=str, help="please input your trt model path")
    parser.add_argument("-f", "--exp_file", default="./selfEXP.py", type=str, help="please input your experiment description file")
    parser.add_argument("-d", "--images_dir", default="./images", type=str, help="Picture folder in directory detection mode")
    parser.add_argument("-o", "--result_path", default=None, type=str, help="result path")
    parser.add_argument("-l", "--mAP_label_result", default="./mAP/input/detection-results", type=str, help="mAP label result path")

    parser.add_argument("-b", "--batch_size", default=8, type=int, help="model detect batch size")
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fp16", type=bool, default=True, help="Adopting mix precision evaluating.")

    return parser.parse_args()


if __name__ == '__main__':
    # args
    args = make_args()

    # 主定义参数
    args.exp_file = "./selfEXP.py"
    args.trt_path = "./models/yolox-m-0608/latest_batch8_upsample_dynamic.fp16.trt"
    args.batch_size = 8
    args.cls_list = ["mask_weared_incorrect", "with_mask", "without_mask"]
    args.result_path = "./result_images"
    args.images_dir = "./mAP/input/images-optional"
    args.mAP_label_result = "./mAP/input/detection-results"

    main_dir_serial_batch(args)