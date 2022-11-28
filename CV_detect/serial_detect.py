import logging
import cv2
import sys

sys.path.append('/home/nvidia/7th_Hackathon/CV_detect')
from detect_utils.pre_process import preprocess_yolov7_batch_images
from detect_utils.trtpy_detect import TRT_Detection
from detect_utils.utils import log_set, get_image_list, make_args
from detect_utils.post_process import post_process_batch


# def get_ratio(img_group: list, exp_size=(640, 640)) -> list:
#     output = [None for _ in range(len(img_group))]
#     exp_height, exp_width = exp_size[0], exp_size[1]
#     for index, img in enumerate(img_group):
#         height, width, channel = img.shape
#         output[index] = min(exp_height / height, exp_width / height)
#     return output


def _main(args):
    # path
    args.result_path = "./result_images"
    args.images_dir = "./mAP/input/images-optional"
    args.result_basename = "test_detect"

    # load engine
    detections = TRT_Detection(
        engine_file_path=args.trt_path,
        cls_list=args.cls_list,
        batch_size=args.batch_size,
        exp_size=(640, 640)
    )

    args.img_path_list = get_image_list(args.images_dir)
    # args.result_output = list()
    # detect
    for index, pre_group in enumerate(preprocess_yolov7_batch_images(
            image_list=[cv2.imread(path) for path in args.img_path_list],
            max_batch=args.batch_size,
            exp_size=(640, 640),
            un_read=False)
    ):
        resized_group, source_group = pre_group[0], pre_group[1]

        # detect
        host_outputs = detections.detect(resized_group)

        # postprocess
        postprocess_result = post_process_batch(host_outputs, batch_size=args.batch_size, conf=args.conf,
                                                nms=args.nms, num_class=len(args.cls_list))
        for num in range(len(postprocess_result)):
            if postprocess_result[num] is not None:
                postprocess_result[num] = postprocess_result[num].tolist()
            else:
                postprocess_result[num] = []
        # args.result_output.append(postprocess_result)

        # visual
        images_path = args.img_path_list[index * args.batch_size: index * args.batch_size + args.batch_size]

    detections.destroy()

    # for index, postprocess_img in enumerate(args.result_output):


if __name__ == '__main__':
    # logging
    log_set(screen_show=logging.DEBUG)

    # args
    args = make_args()

    # basic var
    # args.trt_path = "models/yolov7_rep/yolov7_rep_grid_simplify.fp16.trt"
    args.trt_path = "models/yolov7_tiny/yolov7-tiny-rep.fp16.engine"
    args.batch_size = 8
    args.cls_list = ['CARDBOARD', 'banan', 'bottle']
    args.conf = 0.1
    args.nms = 0.45

    _main(args)
a