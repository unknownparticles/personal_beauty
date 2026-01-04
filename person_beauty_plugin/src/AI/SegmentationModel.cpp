#include "SegmentationModel.h"
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
  if (!engine_.getSession().GetInputCount()) {
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
  std::vector<float> inputTensorValues;
  inputTensorValues.reserve(1 * 3 * inputHeight_ * inputWidth_);

  // HWC to CHW
  std::vector<cv::Mat> channels(3);
  cv::split(resized, channels);
  for (const auto &ch : channels) {
    inputTensorValues.insert(inputTensorValues.end(), ch.begin<float>(),
                             ch.end<float>());
  }

  std::vector<int64_t> inputShape = {1, 3, inputHeight_, inputWidth_};
  Ort::MemoryInfo memoryInfo =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
      memoryInfo, inputTensorValues.data(), inputTensorValues.size(),
      inputShape.data(), inputShape.size());

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
  // Assuming output is NCHW or NHW (mask)
  // For portrait segmentation, usually it produces a mask [1, 1, H, W] or [1,
  // H, W]

  float *floatOutput = outputTensors[0].GetTensorMutableData<float>();
  // Create Mat from output (assuming 256x256 output)
  cv::Mat maskFloat(inputHeight_, inputWidth_, CV_32FC1, floatOutput);

  // Resize back to original
  cv::Mat mask;
  cv::threshold(maskFloat, mask, 0.5, 255, cv::THRESH_BINARY);
  mask.convertTo(mask, CV_8UC1);

  cv::Mat finalMask;
  cv::resize(mask, finalMask, input.getMat().size());

  auto result = std::make_shared<ImageBuffer>(input.getMat().cols,
                                              input.getMat().rows, 1);
  finalMask.copyTo(result->getMat());

  return result;
}

} // namespace AI
} // namespace PersonBeauty
