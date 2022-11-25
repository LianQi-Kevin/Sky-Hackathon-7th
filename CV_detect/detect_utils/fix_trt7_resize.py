import argparse
import os

import onnx


def take_resize_to_Upsample(model, output_path):
    """
    pip install onnx
    如果你是TensorRT7.x，执行时一般会提示Resize节点无法解析，那么这个python文件可以解决解析的问题
    实现方案是把Resize修改为Upsample
    """
    nodes = model.graph.node

    resize_layers = [
        [index, node] for index, node in enumerate(nodes) if node.op_type == "Resize"
    ]

    ignore_constant = []
    for index, layer in resize_layers:
        # will removed
        ignore_constant.append(layer.input[1])

        upsample = onnx.NodeProto()
        attr_mode = onnx.AttributeProto()

        attr_mode.name = "mode"
        attr_mode.s = b"nearest"
        attr_mode.type = onnx.AttributeProto.AttributeType.STRING

        upsample.op_type = "Upsample"
        upsample.name = layer.name
        upsample.input.append(layer.input[0])
        upsample.input.append(layer.input[2])
        upsample.output.append(layer.output[0])
        upsample.attribute.append(attr_mode)
        nodes[index].CopyFrom(upsample)

    for i in range(len(nodes) - 1, -1, -1):
        for oname in nodes[i].output:
            if oname in ignore_constant:
                print("Remove node {}".format(nodes[i].name))
                del nodes[i]
                break

    onnx.save(model, output_path)


if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="change 'Resize' node to 'Upsample'")
    parser.add_argument('--input_path', '-i', type=str, default="yolov5s-6.0.onnx", help='source onnx path')
    parser.add_argument('--output_path', '-o', type=str, default=None, help='output onnx path')
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path

    if output_path is None:
        output_path = "{}_upsample_dynamic.onnx".format(os.path.splitext(input_path)[0])

    # https://github.com/onnx/onnx/blob/v1.2.1/onnx/onnx-ml.proto
    model = onnx.load(input_path)

    take_resize_to_Upsample(model, output_path)
