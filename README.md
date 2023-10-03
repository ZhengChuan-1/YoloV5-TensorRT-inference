# YoloV5-TensorRT-inference
这是一个使用C++推理的tensorRT引擎文件的教程。



## 使用步骤
## 一、onnx文件准备
##### 运行原作者yolov5项目下的export.py，导出所需onnx文件
##### python export.py --weights yolov5s.pt --include onnx


## 二、onnx转engine(trt)文件
##### 使用官方转换器trtexec(文件填写自己对应的路径)
##### sudo ./trtexec --onnx = yolov5s.onnx --saveEngine = yolov5s.trt
##### trtexec文件在TensorRT目录下的bin文件。

