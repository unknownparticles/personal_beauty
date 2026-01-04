#include "FaceLandmarkModel.h"
#include <cstring>
#include <iostream>
#include <opencv2/opencv.hpp>

namespace PersonBeauty {
namespace AI {

FaceLandmarkModel::FaceLandmarkModel() {}

bool FaceLandmarkModel::load(const std::string &modelPath) {
  return engine_.loadModel(modelPath);
}

std::vector<cv::Point2f>
FaceLandmarkModel::getLandmarks(const ImageBuffer &input, const FaceBox &face) {
  if (!engine_.isLoaded())
    return {};

  cv::Mat img = input.getMat();

  // 1. Tight Crop (No padding, direct stretching)
  // Some PFLD models expect the input to be exactly the detection box,
  // even if it stretches the face. This matches the way some exporters work.
  int ix1 = std::max(0, (int)face.x1);
  int iy1 = std::max(0, (int)face.y1);
  int ix2 = std::min(img.cols, (int)face.x2);
  int iy2 = std::min(img.rows, (int)face.y2);

  if (ix2 <= ix1 || iy2 <= iy1)
    return {};

  cv::Rect faceRect(ix1, iy1, ix2 - ix1, iy2 - iy1);
  cv::Mat faceImg = img(faceRect).clone();

  // 2. Preprocess
  cv::Mat resized;
  cv::resize(faceImg, resized, cv::Size(inputSize_, inputSize_));
  cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

  cv::Mat floatImg;
  // Normalization: Many PFLD models use (x - 127.5) / 128.0
  resized.convertTo(floatImg, CV_32F, 1.0f / 128.0f, -127.5f / 128.0f);

  // HWC -> CHW
  std::vector<int64_t> inputDims = {1, 3, inputSize_, inputSize_};
  std::vector<float> inputData(1 * 3 * inputSize_ * inputSize_);
  std::vector<cv::Mat> chans;
  cv::split(floatImg, chans);
  for (int i = 0; i < 3; ++i) {
    std::memcpy(inputData.data() + i * 112 * 112, chans[i].data,
                112 * 112 * sizeof(float));
  }

  // 3. Inference
  std::vector<const char *> inputNames = {"input"};
  std::vector<Ort::Value> inputTensors;
  inputTensors.push_back(Ort::Value::CreateTensor<float>(
      engine_.getMemoryInfo(), inputData.data(), inputData.size(),
      inputDims.data(), inputDims.size()));

  std::vector<const char *> outputNames = {"output"};
  auto outputs = engine_.run(inputNames, inputTensors, outputNames);
  if (outputs.empty())
    return {};

  // 4. Decode
  float *ptr = outputs[0].GetTensorMutableData<float>();
  std::vector<cv::Point2f> points;

  for (int i = 0; i < 68; ++i) {
    float lx = ptr[i * 2];
    float ly = ptr[i * 2 + 1];

    // Map back to global coordinates: x = ix1 + lx * faceRect.width
    // Assuming model output is normalized 0-1 relative to crop
    float global_x = (float)ix1 + lx * faceRect.width;
    float global_y = (float)iy1 + ly * faceRect.height;
    points.push_back(cv::Point2f(global_x, global_y));
  }

  return points;
}

} // namespace AI
} // namespace PersonBeauty
