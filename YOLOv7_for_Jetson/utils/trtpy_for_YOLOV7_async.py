import argparse
import os
import time
import uuid
from math import ceil

import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
from torch import cat as torch_cat
from torch import max as torch_max
from torch.tensor import Tensor
from torchvision.ops import batched_nms
# from yolox.exp import get_exp
# from yolox.utils import vis
from loguru import logger


_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


@logger.catch
class YOLOV7_TRT_Detection(object):
    # def __init__(self, engine_file_path, exp, cls_list, batch_size=1, fp16=False):
    def __init__(self, engine_file_path, cls_list, batch_size=1, fp16=False):
        # basic参数
        self.engine_file_path = engine_file_path
        self.engine = self._load_engine()
        print("Successful load {}".format(os.path.basename(self.engine_file_path)))
        # self.exp = exp
        self.cls_list = cls_list
        self.fp16 = fp16
        self.batch_size = batch_size

        # exp参数
        # self.exp_height = exp.input_size[0]
        # self.exp_width = exp.input_size[1]
        self.exp_height = 416
        self.exp_width = 416
        self.num_classes = 80

        # detect参数
        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        self.context = self._create_context()

    def detect(self, img_resized):
        np.copyto(self.host_inputs[0], img_resized.ravel())
        # 将处理好的图片从CPU内存中复制到GPU显存
        cuda.memcpy_htod_async(
            self.cuda_inputs[0], self.host_inputs[0], self.stream)
        # 开始执行推理任务
        self.context.execute_async(
            batch_size=self.batch_size,
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        # 将推理结果输出从GPU显存复制到CPU内存
        cuda.memcpy_dtoh_async(
            self.host_outputs[0], self.cuda_outputs[0], self.stream)
        # print("detect finish, time: {}".format(time.time()))
        return self.host_outputs[0]

    def visual(self, output, img, cls_conf=0.35):
        if len(output) == 0:
            return img
        # output = np.array(output, dtype=object)
        # ratio = min(self.exp_height / img.shape[0], self.exp_width / img.shape[1])
        # bandboxes = output[:, 0:4]
        # # preprocessing: resize
        # bandboxes /= ratio
        # scores = output[:, 4]
        # classes = output[:, 5]
        bandboxes, scores, classes = self.remapping_result(output, img)

        vis_res = vis(img=img, boxes=bandboxes, scores=scores, cls_ids=classes,
                      conf=cls_conf, class_names=self.cls_list)
        return vis_res

    # 重映射推理结果
    def remapping_result(self, output, img):
        output = np.array(output, dtype=object)
        ratio = min(self.exp_height / img.shape[0], self.exp_width / img.shape[1])
        bandboxes = output[:, 0:4]
        # preprocessing: resize
        bandboxes /= ratio
        scores = output[:, 4]
        classes = output[:, 5]
        return bandboxes, scores, classes

    # 反序列化引擎
    def _load_engine(self):
        assert os.path.exists(self.engine_file_path), "{} not found".format(self.engine_file_path)
        print("Reading engine from file {}".format(self.engine_file_path))
        with open(self.engine_file_path, "rb") as f, trt.Runtime(trt.Logger()) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    # 通过加载的引擎，生成可执行的上下文
    def _create_context(self):
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            # 注意：这里的host_mem需要时用pagelocked memory，以免内存被释放
            host_mem = cuda.pagelocked_empty(size, np.float32)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(cuda_mem))
            if self.engine.binding_is_input(binding):
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)
        return self.engine.create_execution_context()

    def post_process(self, host_outputs, conf=0.3, nms=0.45):
        """
        :param conf:
        :param nms:
        :param host_outputs x, y, w, h, conf, cls1, cls2, cls3, cls4 ······
        :return [[x1, y1, x2, y2, scores, cls_name], [x1, y1, x2, y2, scores, cls_name], ···]
        """

        # xywh2xyxy (4ms)
        # team_num = self.exp.num_classes + 5
        team_num = self.num_classes + 5
        prediction = host_outputs.reshape(int(host_outputs.shape[0] / team_num), team_num)
        box_corner = np.zeros(prediction.shape)
        box_corner[:, 0] = prediction[:, 0] - prediction[:, 2] / 2
        box_corner[:, 1] = prediction[:, 1] - prediction[:, 3] / 2
        box_corner[:, 2] = prediction[:, 0] + prediction[:, 2] / 2
        box_corner[:, 3] = prediction[:, 1] + prediction[:, 3] / 2
        prediction[:, :4] = box_corner[:, :4]
        prediction = Tensor(prediction)

        # get 8400 detections (9ms)
        image_pred = prediction
        class_conf, class_pred = torch_max(image_pred[:, 5: team_num], dim=1, keepdim=True)
        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf).squeeze()
        detections = torch_cat([image_pred[:, :4], image_pred[:, 4].reshape(8400, 1) * class_conf, class_pred.float()],
                               dim=1)
        detections = detections[conf_mask]

        # iou nms (1.49ms)
        nms_out_index = batched_nms(
            boxes=detections[:, :4],
            scores=detections[:, 4],
            idxs=detections[:, 5],
            iou_threshold=nms)
        return detections[nms_out_index]

    def pre_process(self, img: str, swap=(2, 0, 1), un_read: bool = True):
        if un_read:
            ST_time = time.time()
            img = cv2.imread(img, cv2.IMREAD_COLOR)
            print("img read spend: {}".format(time.time() - ST_time))
        if len(img.shape) == 3:
            padded_img = np.ones((self.exp_height, self.exp_width, 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(self.exp.input_size, dtype=np.uint8) * 114

        r = min(self.exp_height / img.shape[0], self.exp_width / img.shape[1])
        resized_img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)),
                                 interpolation=cv2.INTER_LINEAR).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r, img

    def post_process_batch(self, host_outputs, batch_size=1, conf=0.3, nms=0.45, result_path=None):
        """
        :param result_path:
        :param conf:
        :param nms:
        :param host_outputs x, y, w, h, conf, cls1, cls2, cls3, cls4 ······
        :param batch_size:
        :return [[x1, y1, x2, y2, scores, cls_name], [x1, y1, x2, y2, scores, cls_name], ···]
        """
        if result_path is not None:
            np.save("{}.npy".format(self.engine_file_path), host_outputs)

        # xywh2xyxy (4ms)
        # team_num = self.exp.num_classes + 5
        team_num = self.num_classes + 5
        prediction = host_outputs.reshape(batch_size, int(host_outputs.shape[0] / team_num / batch_size), team_num)
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
            # image_pred = prediction
            class_conf, class_pred = torch_max(image_pred[:, 5: team_num], dim=1, keepdim=True)
            conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf).squeeze()
            detections = torch_cat([image_pred[:, :4], image_pred[:, 4].reshape(int(host_outputs.shape[0] / team_num / batch_size), 1) * class_conf, class_pred.float()],
                                   dim=1)
            detections = detections[conf_mask]

            # iou nms
            nms_out_index = batched_nms(
                boxes=detections[:, :4],
                scores=detections[:, 4],
                idxs=detections[:, 5],
                iou_threshold=nms)

            output[i] = detections[nms_out_index]
        return output

    def pre_process_batch(self, image_list, max_batch=1, swap=(2, 0, 1), un_read=False):
        group_num = ceil(len(image_list) / max_batch)
        for num in range(group_num):
            ST_time = time.time()
            output = [np.ones((3, self.exp_height, self.exp_width), dtype=np.float32) * 114 for _ in range(max_batch)]
            for index, img in enumerate(image_list[num * max_batch: (num * max_batch) + max_batch]):
                # once_time = time.time()
                if un_read:
                    # ST_time = time.time()
                    img = cv2.imread(img, cv2.IMREAD_COLOR)
                    # print("img read spend: {}".format(time.time() - ST_time))
                # 创建一个(640, 640, 3)的数组
                padded_img = np.ones((self.exp_height, self.exp_width, 3), dtype=np.uint8) * 114
                # 计算图片实际大小和预期大小插值
                r = min(self.exp_height / img.shape[0], self.exp_width / img.shape[1])
                # resize图片
                resized_img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)),
                                         interpolation=cv2.INTER_LINEAR).astype(np.uint8)
                # 填充resized图片到padded_img
                padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
                # 转换成(3, 640, 640的数组)
                padded_img = padded_img.transpose(swap)
                output[index] = padded_img
                # print("once time: {}".format(time.time() - once_time))
            output = np.array(output)
            # 转换数组位置到内存连续， 加速调用
            print("preprocess batch: {}".format(time.time() - ST_time))
            yield [np.ascontiguousarray(output, dtype=np.float32), image_list[num * max_batch: (num * max_batch) + max_batch]]

    # 释放引擎，释放GPU显存，释放CUDA流
    def __del__(self):
        """Free CUDA memories."""
        del self.stream
        del self.cuda_outputs
        del self.cuda_inputs


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


def main_one(args):
    # MyExp = get_exp(exp_file=args.exp_file)
    detection = YOLOV7_TRT_Detection(
        engine_file_path=args.trt_path,
        # exp=MyExp,
        cls_list=args.cls_list,
        fp16=args.fp16)

    ST_time = time.time()
    img_name = "images/2007_000799.jpg"
    resized_img, _, source_img = detection.pre_process(img_name, un_read=True)
    preprocess_time = time.time()
    print("preprocess: {}".format(preprocess_time - ST_time))
    outputs = detection.detect(img_resized=resized_img)
    detect_time = time.time()
    print("all detect: {}".format(detect_time - preprocess_time))
    prediction = detection.post_process(host_outputs=outputs, conf=0.45, nms=0.45)

    if args.result_path is not None:
        result_path = os.path.join(args.result_path,
                                   "{}_{}".format("yolov7", time.strftime("%Y%m%d-%H%M%S", time.localtime())))
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        vis_img = detection.visual(output=prediction, img=source_img, cls_conf=0.35)
        cv2.imwrite(os.path.join(result_path, os.path.basename(img_name)), vis_img)


def main_dir_async(args):
    # 如果待检测文件夹存在则创建文件列表
    global resized_img
    global basic_num
    assert os.path.exists(args.images_dir), "{} not exists".format(args.images_dir)
    # image_list = get_image_list(args.images_dir)
    image_list = [cv2.imread(img, cv2.IMREAD_COLOR) for img in get_image_list(args.images_dir)]
    print("{} images in {}".format(len(image_list), args.images_dir))
    # print(image_list)

    # 加载检测器
    # MyExp = get_exp(exp_file=args.exp_file)
    detection = YOLOV7_TRT_Detection(
        engine_file_path=args.trt_path,
        # exp=MyExp,
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
        vis_img = detection.visual(output=output["output"], img=output["img"], cls_conf=0.35)
        cv2.imwrite(os.path.join(result_path, "{}.jpg".format(index)), vis_img)


def main_dir_serial(args):
    # 如果待检测文件夹存在则创建文件列表
    assert os.path.exists(args.images_dir), "{} not exists".format(args.images_dir)
    # image_list = get_image_list(args.images_dir)
    image_list = [cv2.imread(img, cv2.IMREAD_COLOR) for img in get_image_list(args.images_dir)]
    print("{} images in {}".format(len(image_list), args.images_dir))

    # 加载检测器
    # MyExp = get_exp(exp_file=args.exp_file)
    detection = YOLOV7_TRT_Detection(
        engine_file_path=args.trt_path,
        # exp=MyExp,
        cls_list=args.cls_list,
        fp16=args.fp16)

    output = []
    async_time = time.time()
    for index, img in enumerate(image_list):
        resized_img, _, source_img = detection.pre_process(img=img, un_read=False)
        host_outputs = detection.detect(resized_img)
        detection.stream.synchronize()
        output.append(detection.post_process(host_outputs))
    print("serial time: {}".format(time.time() - async_time))
    print("output shape: {}".format(len(output)))


def main_dir_serial_batch(args):
    # 如果待检测文件夹存在则创建文件列表
    assert os.path.exists(args.images_dir), "{} not exists".format(args.images_dir)
    assert os.path.exists(args.trt_path), "{} not exists".format(args.trt_path)
    # image_list = get_image_list(args.images_dir)
    image_list = [cv2.imread(img, cv2.IMREAD_COLOR) for img in get_image_list(args.images_dir)]
    print("{} images in {}".format(len(image_list), args.images_dir))

    # 加载检测器
    # MyExp = get_exp(exp_file=args.exp_file)
    detection = YOLOV7_TRT_Detection(
        engine_file_path=args.trt_path,
        # exp=MyExp,
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
    for index, preprocess_return in enumerate(detection.pre_process_batch(image_list, args.batch_size, un_read=False)):
        img_group = preprocess_return[0]
        source_images = preprocess_return[1]
        print(img_group.shape)
        # detect
        ST_time = time.time()
        print("start detect: {} - {}".format(index, ST_time))
        output = detection.detect(img_group)
        np_result_path = os.path.join(result_path, "{}_{}.npy".format(detection.exp.exp_name, index))
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
            vis_img = detection.visual(output=out[a], img=img_group[a], cls_conf=0.35)
            cv2.imwrite(os.path.join(result_path, "{}_{}.jpg".format(num, uuid.uuid4())), vis_img)

    print("async time: {}".format(time.time() - async_time))
    print("output shape: {}".format(len(outputs)))


if __name__ == '__main__':
    # args
    args = make_args()

    # 主定义参数
    args.exp_file = "../selfEXP.py"
    args.trt_path = "../models/yolov7_default.fp16.trt"
    # args.trt_path = "../models/yolox-m/yolox-m-batch8_upsample_dynamic.fp16.trt"
    args.fp16 = True
    args.result_path = "../result"
    args.images_dir = "../test_images"
    args.cls_list = [str(i + 1) for i in range(80)]
    # print(args.cls_list)

    # main_one(args)
    # main_dir_async(args)
    # main_dir_serial(args)
    # main_dir_serial_batch(args)
