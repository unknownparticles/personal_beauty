#include "SegmentationModel.h"
#include <algorithm>
#include <cstring>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace PersonBeauty {
namespace AI {

SegmentationModel::SegmentationModel() {}

bool SegmentationModel::load(const std::string &modelPath) {
  return engine_.loadModel(modelPath);
}

std::shared_ptr<ImageBuffer>
SegmentationModel::process(const ImageBuffer &input) {
  if (!engine_.isLoaded()) {
    return nullptr;
  }

  // 1. Preprocess
  cv::Mat resized;
  cv::resize(input.getMat(), resized, cv::Size(inputWidth_, inputHeight_));

  // Normalize and convert to float CHW (Planar)
  // This is a simplified example. Real models might need specific normalization
  // (mean/std)
  resized.convertTo(resized, CV_32F, 1.0f / 255.0f);

  // Prepare input tensor
  // ONNX Runtime expects NCHW float array
  const int area = inputHeight_ * inputWidth_;
  std::vector<float> inputTensorValues(3 * area);

  // HWC to CHW
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
  // Assuming 1 input and 1 output for now.
  // In production, get names from session metadata
  const char *inputNames[] = {"input"};
  const char *outputNames[] = {"output"};

  std::vector<Ort::Value> inputTensors;
  inputTensors.push_back(std::move(inputTensor));

  auto outputTensors =
      engine_.run({inputNames[0]}, inputTensors, {outputNames[0]});

  if (outputTensors.empty())
    return nullptr;

  // 3. Postprocess
  const auto &tensor = outputTensors[0];
  auto typeInfo = tensor.GetTensorTypeAndShapeInfo();
  auto shape = typeInfo.GetShape();
  if (shape.size() < 3) {
    return nullptr;
  }

  const int64_t n = shape[0];
  const int64_t c = shape.size() == 4 ? shape[1] : 1;
  const int64_t h = shape.size() == 4 ? shape[2] : shape[1];
  const int64_t w = shape.size() == 4 ? shape[3] : shape[2];
  if (n != 1 || h <= 0 || w <= 0) {
    return nullptr;
  }

  const size_t outArea = static_cast<size_t>(h * w);
  const float *floatOutput = tensor.GetTensorData<float>();
  std::vector<uint8_t> maskData(outArea, 0);

  if (c > 1) {
    // 多通道语义分割，取 argmax 后认为非背景(>0)即前景
    for (size_t idx = 0; idx < outArea; ++idx) {
      int bestClass = 0;
      float bestVal = floatOutput[idx];
      for (int cls = 1; cls < c; ++cls) {
        float val = floatOutput[cls * outArea + idx];
        if (val > bestVal) {
          bestVal = val;
          bestClass = cls;
        }
      }
      maskData[idx] = bestClass > 0 ? 255 : 0;
    }
  } else {
    // 单通道概率，0.5 阈值
    for (size_t idx = 0; idx < outArea; ++idx) {
      maskData[idx] = floatOutput[idx] >= 0.5f ? 255 : 0;
    }
  }

  cv::Mat coarseMask(h, w, CV_8UC1, maskData.data());
  cv::Mat finalMask;
  cv::resize(coarseMask, finalMask, input.getMat().size(), 0, 0,
             cv::INTER_NEAREST);

  auto result =
      std::make_shared<ImageBuffer>(input.getMat().cols, input.getMat().rows, 1);
  finalMask.copyTo(result->getMat());

  return result;
}

} // namespace AI
} // namespace PersonBeauty
