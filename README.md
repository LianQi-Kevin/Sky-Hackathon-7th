### YOLOV7 ON Jetson Nano
本项目是为了参加 NVIDIA Sky-Hackathon 7th 而建立。

其分为三个部分：CV 、ASR 和 WebUI，CV部分实现了mAP和FPS的获取，项目在 Jetson Nano 上完成了其全部的部署和测试，采用将`预处理`、`检测`、`后处理`三个阶段封装到多线程以进行提速。

#### Step 1. -  CV
Folder: `CV_detect`
1. `serial_detect.py`: 串行推理
2. `multi_thread_detect.py`: 多线程推理, 封装各阶段函数到子线程。
3. `detect_utils/trtpy_detect.py`: TRT_Detection 类，完成模型加载等操作
4. `detect_utils/pre_process.py`: YOLOX 和 YOLOV7 的预处理函数
5. `detect_utils/post_process.py`: 后处理函数, 使用 numpy 完成 NMS 操作，
   代换掉 torchvision.ops.batched_nms 函数。
   使之不依赖于 torch 完成，但是相较其有速度损失。
   
   注: 此函数可以替换掉 TRT_Detection 类的 post_process 函数，输入和输出均一致  

6. `detect_utils/utils.py`: 一些小工具


---
> ##### Note
> 1. 2022-11-22. Gradio can't used in Jetson Nano default python3.6 environments, it needs at least python>=3.7