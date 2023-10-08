#include<fstream>  
#include<iostream> 
#include<opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include "processing.hpp"

#include "NvInfer.h"
#include "logging.h"
using namespace nvinfer1;
using namespace std;


class MyLogger : public nvinfer1::ILogger {
 public:
  explicit MyLogger(nvinfer1::ILogger::Severity severity =
                        nvinfer1::ILogger::Severity::kWARNING)
      : severity_(severity) {}

  void log(nvinfer1::ILogger::Severity severity,
           const char *msg) noexcept override {
    if (severity <= severity_) {
      std::cerr << msg << std::endl;
    }
  }
  nvinfer1::ILogger::Severity severity_;
};

int main(int argc, char* argv[])
{
    string video_path = argv[1];
    string save_path = argv[2];
//-------------------------- 一、定义推理模型------------------------------------------
    MyLogger logger;
    //读取trt信息
    const std::string engine_file_path = "/home/zc/C++_TensorRT_inference/sample_video/yolov5s.trt";  //填写自己trt文件路径(需要绝对路径)
    std::stringstream engine_file_stream;
    engine_file_stream.seekg(0, engine_file_stream.beg);  //从起始位置偏移0个字节，指针移动到文件流的开头
    std::ifstream ifs(engine_file_path);
    engine_file_stream << ifs.rdbuf();
    ifs.close();

    engine_file_stream.seekg(0, std::ios::end);         //先把文件输入流指针定位到文档末尾来获取文档的长度
    const int model_size = engine_file_stream.tellg();  //获取文件流的总长度
    engine_file_stream.seekg(0, std::ios::beg);
    void *model_mem = malloc(model_size);               //开辟一样长的空间
    engine_file_stream.read(static_cast<char *>(model_mem), model_size);    //将内容读取到model_mem中

    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(logger);
    nvinfer1::ICudaEngine *engine = runtime->deserializeCudaEngine(model_mem, model_size);

    free(model_mem);

    nvinfer1::IExecutionContext *context = engine->createExecutionContext();

    void *buffers[2];
    // 获取模型输入尺寸并分配GPU内存
    nvinfer1::Dims input_dim = engine->getBindingDimensions(0);
    int input_size = 1;
    for (int j = 0; j < input_dim.nbDims; ++j) {
        input_size *= input_dim.d[j];
    }
    cudaMalloc(&buffers[0], input_size * sizeof(float));

    // 获取模型输出尺寸并分配GPU内存
    nvinfer1::Dims output_dim = engine->getBindingDimensions(1);
    int output_size = 1;
    for (int j = 0; j < output_dim.nbDims; ++j) {
        output_size *= output_dim.d[j];
    }
    cudaMalloc(&buffers[1], output_size * sizeof(float));

    // 给模型输出数据分配相应的CPU内存
    float *output_buffer = new float[output_size];

    cudaStream_t stream;
    cudaStreamCreate(&stream);
//-----------------------------------------------------------------------

    const int model_width = 640;
    const int model_height = 640;
    float* input_blob = new float[model_height * model_width * 3];

    cv::VideoCapture cap(video_path);

    int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
	int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	int frame_count = cap.get(cv::CAP_PROP_FRAME_COUNT);
	double fps = cap.get(cv::CAP_PROP_FPS);
    //第1个参数 视频文件路径；第2个参数 视频编码方式（我们可以通过VideoCapture::get(CAP_PROP_FOURCC)获得）；第3个参数 fps；第4个参数 size；第5个参数 是否为彩色
    //cv::VideoWriter write(save_path, cap.get(cv::CAP_PROP_FOURCC), fps, cv::Size(frame_width, frame_height), true);
    cv::VideoWriter write(save_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(frame_width, frame_height), true);
    cv::Mat frame;
    cv::Mat resize_image;
    while(true){
        int ret = cap.read(frame);  //frame就是每一帧图片
        if(!ret){break;}
        
//----------------- 二、图像预处理  --------------------------------------
        const float ratio = std::min(model_width / (frame.cols * 1.0f),
                                model_height / (frame.rows * 1.0f));
        // 等比例缩放
        const int border_width = frame.cols * ratio;
        const int border_height = frame.rows * ratio;
        // 计算偏移值
        const int x_offset = (model_width - border_width) / 2;
        const int y_offset = (model_height - border_height) / 2;

        //将输入图像缩放至resize_image
        cv::resize(frame, resize_image, cv::Size(border_width, border_height));
        //复制图像并且制作边界
        cv::copyMakeBorder(resize_image, resize_image, y_offset, y_offset, x_offset,
                            x_offset, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
        // 转换为RGB格式
        cv::cvtColor(resize_image, resize_image, cv::COLOR_BGR2RGB);
        //归一化
        normalization(resize_image, input_blob);
//---------------------------------------------------------------------

//-------------三、往engine引擎投入预处理后图像----------------------------
        // 拷贝输入数据至GPU
        cudaMemcpyAsync(buffers[0], input_blob, input_size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
        // 执行推理
        if(!context->enqueueV2(buffers, stream, nullptr))
        {
            cout << "enqueueV2执行推理失败" << endl;
            return false;
        }
        // 拷贝输出数据至CPU
        cudaMemcpyAsync(output_buffer, buffers[1],output_size * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
        // 使同步
        cudaStreamSynchronize(stream);
//---------------------------------------------------------------------

//---------------四、预测结果(output_buffer)后处理------------------------
        postprocessing(output_buffer, frame, x_offset, y_offset, ratio);
//---------------------------------------------------------------------

        write.write(frame);
        cv::imshow("video", frame);

        int c = cv::waitKey(1);
		if (c == 27) {break;}    //判断若按ESC键退出循环

    }

    delete context;
    delete engine;
    delete runtime;
    delete[] input_blob;

    cap.release();
    write.release();

    return 0;
}