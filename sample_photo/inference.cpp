#include<fstream>  
#include<iostream>  
#include<opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include "processing.hpp"
#include <iomanip>  //保留小数

#include "NvInfer.h"
#include "logging.h"

using namespace std;
using namespace nvinfer1;

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

int main()
{
//一、图像处理
    const int model_width = 640;
    const int model_height = 640;

    //string image_path = "/home/zc/C++_TensorRT_inference/sample_photo/images/bus.jpg";
    string image_path = "/home/zc/C++_TensorRT_inference/sample_photo/images/zidane.jpg";  //填写自己图片路径(需要绝对路径)
    float* input_blob = new float[model_height * model_width * 3];

    cv::Mat input_image = cv::imread(image_path);
    cv::Mat resize_image;

    const float ratio = std::min(model_width / (input_image.cols * 1.0f),
                            model_height / (input_image.rows * 1.0f));
    // 等比例缩放
    const int border_width = input_image.cols * ratio;
    const int border_height = input_image.rows * ratio;
    // 计算偏移值
    const int x_offset = (model_width - border_width) / 2;
    const int y_offset = (model_height - border_height) / 2;

    //将输入图像缩放至resize_image
    cv::resize(input_image, resize_image, cv::Size(border_width, border_height));
    //复制图像并且制作边界
    cv::copyMakeBorder(resize_image, resize_image, y_offset, y_offset, x_offset,
                        x_offset, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    // 转换为RGB格式
    cv::cvtColor(resize_image, resize_image, cv::COLOR_BGR2RGB);
    
    //归一化
    const int channels = resize_image.channels();
    const int width = resize_image.cols;
    const int height = resize_image.rows;
    for (int c = 0; c < channels; c++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                input_blob[c * width * height + h * width + w] =
                    resize_image.at<cv::Vec3b>(h, w)[c] / 255.0f;  //at<Vec3b> 是 OpenCV 中用于访问图像像素的一种方法，使用 at<Vec3b> 获取彩色图像中特定位置的像素颜色值
            }
        }
    }

//二、模型反序列化
    MyLogger logger;
    //读取trt信息
    const std::string engine_file_path = "/home/zc/C++_TensorRT_inference/sample_photo/yolov5s.trt";  //填写自己trt文件路径(需要绝对路径)
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

//三、模型推理
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

    //投入数据
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    // 拷贝输入数据
    cudaMemcpyAsync(buffers[0], input_blob,input_size * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    // 执行推理
    if(context->enqueueV2(buffers, stream, nullptr))
    {
        cout << "enqueueV2执行推理成功" << endl;
    }
    else{
        cout << "enqueueV2执行推理失败" << endl;
        return -1;
    }
    // 拷贝输出数据
    cudaMemcpyAsync(output_buffer, buffers[1],output_size * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    delete context;
    delete engine;
    delete runtime;
    delete[] input_blob;

//四、输出结果output_buffer，放入objs  xywh为中心点坐标 和宽高
    float *ptr = output_buffer;
    std::vector<Object> objs;
    for (int i = 0; i < 25200; ++i) {
        const float objectness = ptr[4];
        if (objectness >= 0.45f) {
            const int label = std::max_element(ptr + 5, ptr + 85) - (ptr + 5);  //std::max_element返回范围内的最大元素
            const float confidence = ptr[5 + label] * objectness;
            if (confidence >= 0.25f) {
                const float bx = ptr[0];
                const float by = ptr[1];
                const float bw = ptr[2];
                const float bh = ptr[3];

                Object obj;
                // 还原图像尺寸中box的尺寸比例，这里要减掉偏移值，并把box中心点坐标xy转成左上角坐标xy
                obj.box.x = (bx - bw * 0.5f - x_offset) / ratio;
                obj.box.y = (by - bh * 0.5f - y_offset) / ratio;
                obj.box.width = bw / ratio;
                obj.box.height = bh / ratio;
                obj.label = label;
                obj.confidence = confidence;
                objs.push_back(std::move(obj));
                }
        }
        ptr += 85;
    }  // i loop

//五、NMS非极大值抑制
    vector<list<Object>> finalll = NMS(objs);

//六、画框
    int row = sizeof(finalll) / sizeof(finalll[0]);
    for(int i = 0; i <= row; i++){
        list<Object>::iterator it = finalll[i].begin();
        while(it != finalll[i].end()){
            cv::Point topLeft(it->box.x, it->box.y);
            cv::Point bottomRight(it->box.x + it->box.width, it->box.y + it->box.height);
            cv::rectangle(input_image, topLeft, bottomRight, cv::Scalar(0, 0, 255), 2);
            std::stringstream buff;
            buff.precision(2);  //覆盖默认精度,置信度保留2位小数
            buff.setf(std::ios::fixed);
            buff << it->confidence;
            string text =names[it->label] + " " + buff.str();
            cv::putText(input_image, text, topLeft, 0, 1, cv::Scalar(0, 255, 0), 2);
            it++;
        }
    }
    //cv::imwrite("_bus.jpg", input_image);
    cv::imwrite("_zidane.jpg", input_image);
    return 0;
}

