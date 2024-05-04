//
// Created by 蔡雄江 on 2024/4/11.
//

#ifndef MODEL_TRT_CPP_MODEL_INFER_H
#define MODEL_TRT_CPP_MODEL_INFER_H

#include <map>

#include <NvInfer.h> 

using namespace nvinfer1; 
using namespace sample; 


#define CHECK(status) \ 
    do\ 
    {\ 
        auto ret = (status);\ 
        if (ret != 0)\ 
        {\ 
            std::cerr << "Cuda failure: " << ret << std::endl;\ 
            abort();\ 
        }\ 
    } while (0) 


class classifyModel {
public:
    classifyModel(const std::string modelName);
    ~classifyModel();

    std::string getEngineFile(std::string modelName);
    float* preProcess(const string& image_path);
    void Inference(IExecutionContext& context, float* input, float* output, int batchSize);
    void postProcess();

    std::string EngineFile;


private:
    std::vector<std::string> modelList = {
        "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
        "vit-t", "vit-s", "vit-l"
    }; 

    const char* IN_NAME = "input"; 
    const char* OUT_NAME = "output"; 
    static const int IN_H = 224; 
    static const int IN_W = 224; 
    static const int BATCH_SIZE = 1; 
    static const int EXPLICIT_BATCH = 1 << (int)(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH); 

    IExecutionContext* context;
};


#endif //MODEL_TRT_CPP_MODEL_INFER_H
