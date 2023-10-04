# YoloV5-TensorRT-inference
这是一个使用C++推理的tensorRT引擎文件的案例。<br/><br/>
系统：Ubuntu 编译：VScode + Cmake 
<br/><br/>
## <font color=“blue”>使用步骤</font>
## 一、onnx文件准备
##### 运行原作者yolov5项目下的export.py，导出所需onnx文件
##### python export.py --weights yolov5s.pt --include onnx
<br/><br/>
## 二、onnx转engine(trt)文件
##### 使用官方转换器trtexec(文件填写自己对应的路径)
##### sudo ./trtexec --onnx = yolov5s.onnx --saveEngine = yolov5s.trt
##### trtexec文件在TensorRT目录下的bin文件。
<br/><br/>
## 三、Cmake
##### CMakeLists.txt 路径依赖替换成自己的路径
##### cd ./build/
##### cmake ..
##### make
##### 现在build文件夹中生成一个可执行文件inference
##### ./inference 运行即刻完成推理输出
