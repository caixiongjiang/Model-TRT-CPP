//
// Created by 蔡雄江 on 2024/4/11.
//

#include <iostream> 
#include <vector>

#include <opencv2/opencv.hpp>

#include "model_infer.h"

using namespace nvinfer1; 


classifyModel::classifyModel(const std::string modelName) {
    // 检查模型文件是否存在
    if (std::find(modelList.begin(), modelList.end(), modelName) != modelList.end()) {
        std::cerr << "Error: modelName should be in ";
        for (const auto& model : modelList) {
            std::cerr << model << " ";
        }
        std::cerr << std::endl;
        exit(EXIT_FAILURE); // 终止程序并返回失败状态
    }
    // 加载Engine
    char *trtModelStream{ nullptr }; 
    size_t size{ 0 }; 

    std::ifstream file("model.engine", std::ios::binary); 
    if (file.good()) { 
        file.seekg(0, file.end); 
        size = file.tellg(); 
        file.seekg(0, file.beg); 
        trtModelStream = new char[size]; 
        assert(trtModelStream); 
        file.read(trtModelStream, size); 
        file.close(); 
    } 

    Logger m_logger; 
    IRuntime* runtime = createInferRuntime(m_logger); 
    assert(runtime != nullptr); 
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr); 
    assert(engine != nullptr); 
    context = engine->createExecutionContext(); 
    assert(context != nullptr);
}



std::string classifyModel::getEngineFile(std::string modelName) {
    std::string EngineFilePath = "./data/model_zoo/classification/" + modelName + ".trt";
    return EngineFilePath;
}


float* classifyModel::preProcess(const string& image_path) {
    // Load the image using OpenCV
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Error: Unable to load image from path " << image_path << std::endl;
        return;
    }

    // Resize the image to the desired size (224x224)
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(IN_W, IN_H));

    // Convert the image to float and normalize it to [0, 1]
    resized_image.convertTo(resized_image, CV_32FC3, 1.0 / 255.0);

    std::vector<cv::Mat> channels;
    cv::split(resized_image, channels);

    float* input = new float[3 * IN_H * IN_W];

    int channel_size = IN_H * IN_W;
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < IN_H; ++i) {
            for (int j = 0; j < IN_W; ++j) {
                input[c * channel_size + i * IN_W + j] = channels[c].at<float>(i, j);
            }
        }
    }

    return input;
}

void classifyModel::Inference(IExecutionContext& context, float* input, float* output, int batchSize) {
    const ICudaEngine& engine = context.getEngine(); 

    // Pointers to input and output device buffers to pass to engine. 
    // Engine requires exactly IEngine::getNbBindings() number of buffers. 
    assert(engine.getNbBindings() == 2); 
    void* buffers[2]; 

    // In order to bind the buffers, we need to know the names of the input and output tensors. 
    // Note that indices are guaranteed to be less than IEngine::getNbBindings() 
    const int inputIndex = engine.getBindingIndex(IN_NAME); 
    const int outputIndex = engine.getBindingIndex(OUT_NAME); 

    // Create GPU buffers on device 
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * IN_H * IN_W * sizeof(float))); 
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * 3 * IN_H * IN_W /4 * sizeof(float))); 

    // Create stream 
    cudaStream_t stream; 
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host 
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * IN_H * IN_W * sizeof(float), cudaMemcpyHostToDevice, stream)); 
    context.enqueue(batchSize, buffers, stream, nullptr); 
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * 3 * IN_H * IN_W / 4 * sizeof(float), cudaMemcpyDeviceToHost, stream)); 
    cudaStreamSynchronize(stream); 

    // Release stream and buffers 
    cudaStreamDestroy(stream); 
    CHECK(cudaFree(buffers[inputIndex])); 
    CHECK(cudaFree(buffers[outputIndex])); 
}

void classifyModel::postProcess() {
    
}