#include "ParsingModel.h"
#include <algorithm>
#include <cstring>
#include <opencv2/imgproc.hpp>

namespace PersonBeauty {
namespace AI {

ParsingModel::ParsingModel() {}

bool ParsingModel::load(const std::string &modelPath) {
  return engine_.loadModel(modelPath);
}

std::shared_ptr<ImageBuffer> ParsingModel::process(const ImageBuffer &input) {
  if (!engine_.isLoaded())
    return nullptr;

  // 1. Preprocess
  cv::Mat resized;
  cv::resize(input.getMat(), resized, cv::Size(inputWidth_, inputHeight_));

  // Normalization (Mean/Std often needed for parsing models like BiSeNet)
  // Here using a generic placeholder normalization
  resized.convertTo(resized, CV_32F, 1.0f / 255.0f);
  cv::subtract(resized, cv::Scalar(0.485, 0.456, 0.406), resized);
  cv::divide(resized, cv::Scalar(0.229, 0.224, 0.225), resized);

  // Prepare Tensor
  const int area = inputHeight_ * inputWidth_;
  std::vector<float> inputTensorValues(3 * area);

  std::vector<cv::Mat> channels(3);
  cv::split(resized, channels);
  for (int c = 0; c < 3; ++c) {
    std::memcpy(inputTensorValues.data() + c * area, channels[c].ptr<float>(),
                area * sizeof(float));
  }

  std::vector<int64_t> inputShape = {1, 3, inputHeight_, inputWidth_};
  Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
      engine_.getMemoryInfo(), inputTensorValues.data(),
      inputTensorValues.size(), inputShape.data(), inputShape.size());

  // 2. Run Inference
  const char *inputNames[] = {"input"};
  const char *outputNames[] = {"output"};

  std::vector<Ort::Value> inputTensors;
  inputTensors.push_back(std::move(inputTensor));

  auto outputTensors =
      engine_.run({inputNames[0]}, inputTensors, {outputNames[0]});

  if (outputTensors.empty())
    return nullptr;

  // 3. Postprocess (ArgMax)
  // Output is usually [1, NumClasses, H, W] -> We want [H, W] with class
  // indices

  float *floatOutput = outputTensors[0].GetTensorMutableData<float>();
  // Assuming 19 classes
  int numClasses = 19;
  // Usually parsing output is same resolution as input or 1/8th -> resize up
  // Let's assume output is 512x512

  cv::Mat parsingMap(inputHeight_, inputWidth_, CV_8UC1);

  // Efficient ArgMax (this is slow in CPU loop, but standard for C++ demo)
  int area = inputHeight_ * inputWidth_;
  for (int i = 0; i < area; ++i) {
    int maxClass = 0;
    float maxVal = floatOutput[i]; // Class 0

    for (int c = 1; c < numClasses; ++c) {
      float val = floatOutput[c * area + i];
      if (val > maxVal) {
        maxVal = val;
        maxClass = c;
      }
    }
    parsingMap.data[i] = static_cast<uchar>(maxClass);
  }

  // Resize back to original size (Nearest Neighbor to keep class IDs)
  cv::Mat finalMask;
  cv::resize(parsingMap, finalMask, input.getMat().size(), 0, 0,
             cv::INTER_NEAREST);

  auto result = std::make_shared<ImageBuffer>(input.getMat().cols,
                                              input.getMat().rows, 1);
  finalMask.copyTo(result->getMat());

  return result;
}

} // namespace AI
} // namespace PersonBeauty
