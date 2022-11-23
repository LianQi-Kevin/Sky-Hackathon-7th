import argparse
import os
import time
import uuid
import cv2
from trtpy_for_YOLOV7_async import YOLOV7_TRT_Detection


def make_args():
    parser = argparse.ArgumentParser("YOLOX TRT DETECT!")
    parser.add_argument("-tp", "--trt_path", default=None, type=str,
                        help="please input your trt model path")
    parser.add_argument("-f", "--exp_file", default=None, type=str,
                        help="please input your experiment description file")
    parser.add_argument("-d", "--images_dir", default="./images", type=str,
                        help="Picture folder in directory detection mode")
    parser.add_argument("-b", "--batch_size", default=1, type=int,
                        help="model detect batch size")
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fp16", type=bool, default=False,
                        help="Adopting mix precision evaluating.")
    parser.add_argument("-o", "--result_path", default=None, type=str,
                        help="result path")

    return parser.parse_args()


def get_image_list(path):
    image_names = []
    for main_dir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            a_path = os.path.join(main_dir, filename)
            ext = os.path.splitext(a_path)[1]
            if ext in [".jpg", ".jpeg", ".webp", ".bmp", ".png"]:
                image_names.append(a_path)
    return image_names


def main_dir_async_batch(args):
    # 如果待检测文件夹存在则创建文件列表
    global resized_img
    global basic_num
    assert os.path.exists(args.images_dir), "{} not exists".format(args.images_dir)
    # image_list = get_image_list(args.images_dir)
    image_list = [cv2.imread(img, cv2.IMREAD_COLOR) for img in get_image_list(args.images_dir)]
    print("{} images in {}".format(len(image_list), args.images_dir))

    # 加载检测器
    detection = YOLOV7_TRT_Detection(
        engine_file_path=args.trt_path,
        cls_list=args.cls_list,
        fp16=args.fp16)

    basic_num = 0
    outputs = {}
    async_time = time.time()
    for index, img in enumerate(image_list):
        # preprocess
        resized_img, _, source_img = detection.pre_process(img=img, un_read=False)
        print("finish preprocess: {} - {}".format(basic_num, time.strftime("%Y%m%d-%H%M%S", time.localtime())))
        outputs[index] = {"img": source_img}
        if index != 0:
            detection.stream.synchronize()
            print("finish detect: {} - {}".format(basic_num - 1, time.strftime("%Y%m%d-%H%M%S", time.localtime())))
            basic_num += 1

        # detect
        print("start detect: {} - {}".format(basic_num, time.localtime()))
        detection.detect(resized_img)
        time.sleep(0.005)
        if index == 0 or img == image_list[-1]:
            detection.stream.synchronize()
            print("finish detect: {} - {}".format(basic_num, time.strftime("%Y%m%d-%H%M%S", time.localtime())))
            basic_num += 1

        # postprocess
        if basic_num - 1 == index or img == image_list[-1]:
            output = detection.post_process(detection.host_outputs[0])
            print("finish postprocess: {} - {}".format(basic_num, time.strftime("%Y%m%d-%H%M%S", time.localtime())))
            print(output)

            outputs[index]["output"] = output
        print("once time: {}".format(time.strftime("%Y%m%d-%H%M%S", time.localtime())))

    print("async time: {}".format(time.time() - async_time))
    print("output shape: {}".format(len(outputs)))
    # exit()
    # draw output
    result_path = os.path.join(args.result_path,
                               "{}_{}".format("yolov7", time.strftime("%Y%m%d-%H%M%S", time.localtime())))
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    for index, output in enumerate(outputs.values()):
        vis_img = detection.visual(output=output["output"], img=output["img"], cls_conf=0.8)
        cv2.imwrite(os.path.join(result_path, "{}.jpg".format(index)), vis_img)


def main_dir_serial_batch(args):
    # 如果待检测文件夹存在则创建文件列表
    assert os.path.exists(args.images_dir), "{} not exists".format(args.images_dir)
    assert os.path.exists(args.trt_path), "{} not exists".format(args.trt_path)
    # image_list = get_image_list(args.images_dir)
    image_list = [cv2.imread(img, cv2.IMREAD_COLOR) for img in get_image_list(args.images_dir)]
    print("{} images in {}".format(len(image_list), args.images_dir))

    # 加载检测器
    detection = YOLOV7_TRT_Detection(
        engine_file_path=args.trt_path,
        cls_list=args.cls_list,
        batch_size=args.batch_size,
        fp16=args.fp16)

    # draw output
    result_path = os.path.join(args.result_path,
                               "{}_{}".format("yolov7", time.strftime("%Y%m%d-%H%M%S", time.localtime())))
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    outputs = []
    async_time = time.time()
    for index, preprocess_return in enumerate(detection.pre_process_batch_yolov7(image_list, args.batch_size, un_read=False)):
        img_group = preprocess_return[0]
        source_images = preprocess_return[1]
        # detect
        ST_time = time.time()
        print("start detect: {} - {}".format(index, ST_time))
        output = detection.detect(img_group)
        np_result_path = os.path.join(result_path, "{}_{}.npy".format("yolov7", index))
        detection.stream.synchronize()
        FINDET_time = time.time()
        print("finish detect {} - {}".format(index, FINDET_time - ST_time))
        output = detection.post_process_batch(host_outputs=output, batch_size=args.batch_size,
                                              result_path=np_result_path)
        print("post process: {} - {}".format(index, time.time() - FINDET_time))
        outputs.append([[out.tolist() for out in output], source_images])
        print([out.tolist() for out in output])
        # pass
        print("once time: {}".format(time.time() - ST_time))

    for num, output in enumerate(outputs):
        out = output[0]
        img_group = output[1]
        for a in range(len(img_group)):
            vis_img = detection.visual(output=out[a], img=img_group[a], cls_conf=0.6)
            cv2.imwrite(os.path.join(result_path, "{}_{}.jpg".format(num, uuid.uuid4())), vis_img)

    print("async time: {}".format(time.time() - async_time))
    print("output shape: {}".format(len(outputs)))


if __name__ == '__main__':
    # args
    args = make_args()

    # 主定义参数
    # args.exp_file = "../selfEXP.py"
    args.trt_path = "../models/01/yolov7_default.fp16.trt"
    args.fp16 = True
    args.result_path = "../result"
    args.images_dir = "../infer_images"
    args.cls_list = ["banana", "CARDBOARD", "bottle"]

    main_dir_serial_batch(args)
