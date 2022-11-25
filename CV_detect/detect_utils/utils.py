import logging
import os
import argparse
from glob import glob


# logging set
def log_set(file_write=logging.INFO, screen_show=logging.DEBUG):
    logger = logging.getLogger()  # 不加名称设置root logger
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # 使用FileHandler输出到文件
    fh = logging.FileHandler('log.log')
    fh.setLevel(file_write)
    fh.setFormatter(formatter)

    # 使用StreamHandler输出到屏幕
    ch = logging.StreamHandler()
    ch.setLevel(screen_show)
    ch.setFormatter(formatter)

    # 添加两个Handler
    logger.addHandler(ch)
    logger.addHandler(fh)


# get all image files in path
def get_image_list(path: str) -> list:
    return glob(os.path.join(path, "*[.jpg, .jpeg, .webp, .bmp, .png]"))


# basic args
def make_args():
    parser = argparse.ArgumentParser("YOLOX TRT DETECT!")
    parser.add_argument("-tp", "--trt_path", default=None, type=str,
                        help="trt model path")
    parser.add_argument("-d", "--images_dir", default="./images", type=str,
                        help="Picture folder in directory detection mode")
    parser.add_argument("-b", "--batch_size", default=1, type=int,
                        help="model detect batch size")
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("-o", "--result_path", default=None, type=str, help="result path")
    return parser.parse_args()


# result to mAP format
def to_mAP(bandboxes: list, scores: list, classes: list, class_list: list) -> list:
    """
    :param bandboxes: remapped bandboxes
    :param scores: remapped scores
    :param classes: remapped classes
    :param class_list: each category names list
    :return [[category, x1, y1, x2, y2], [category, x1, y1, x2, y2], ...]
    """
    outputs = list()
    for box, score, class_id in zip(bandboxes, scores, classes):
        class_name = class_list[int(class_id)]
        outputs.append([class_name, score,
                        int(box[0]), int(box[1]), int(box[2]), int(box[3])])
    return outputs


