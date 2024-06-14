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
##### sudo ./trtexec --onnx=yolov5s.onnx --saveEngine=yolov5s.trt
##### trtexec文件在TensorRT目录下的bin文件。

yolov5s.trt文件百度云下载：链接: https://pan.baidu.com/s/1nmrxr7n19ZxwldBJw64USA?pwd=QWER 提取码: QWER
## 三、Cmake
##### CMakeLists.txt 路径依赖替换成自己的路径
##### 代码中图片、engine(trt)文件、保存路径 替换成自己路径
##### VScode项目中按ctrl+shift+P，选择Cmake配置，此时生成build文件夹 
##### cd ./build/
##### cmake ..
##### make
##### 现在build文件夹中生成一个可执行文件infer
##### 在sample_photo中：
##### 命令行输入./infer 运行即可完成推理输出

##### 在sample_video中：命令行输入需要添加视频源文件路径和输出路径两个参数
##### 例： ./infer /dev/video0 /home/zc/C++_TensorRT_inference/sample_video/video/0_0.mp4
#####     ./infer /home/zc/C++_TensorRT_inference/sample_video/video/0.mp4 /home/zc/C++_TensorRT_inference/sample_video/video/0_0.mp4
