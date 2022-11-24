import os
import argparse


def make_parser():
    parser = argparse.ArgumentParser("TensorRT acceleration")
    parser.add_argument("--input_file", "-i", type=str, default="yolox.onnx", help="output name of models")
    parser.add_argument("--batch_size", "-b", type=int, default=8, help="batch size")
    parser.add_argument("--fp16", action="store_true", help="fp16")

    return parser.parse_args()


def onnx_to_trt(onnx_file, max_batch_size=8, fp16=True):
    output_file = onnx_file.replace(".onnx", ".trt")

    if fp16:
        fp16 = 16
        output_file = output_file.replace(".trt", ".fp16.trt")
    else:
        fp16 = 32

    if os.path.exists(onnx_file):
        os.system("onnx2trt {} -o {}  -b {} -d {}".format(onnx_file, output_file, max_batch_size, fp16))


if __name__ == '__main__':
    args = make_parser()
    assert os.path.exists(args.input_file), "{} not found".format(args.input_file)

    # model2trt
    onnx_to_trt(onnx_file=args.input_file, max_batch_size=args.batch_size, fp16=args.fp16)
