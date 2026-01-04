#include "FaceLandmarkModel.h"
#include <iostream>
#include <opencv2/opencv.hpp>

namespace PersonBeauty {
namespace AI {

FaceLandmarkModel::FaceLandmarkModel() {}

bool FaceLandmarkModel::load(const std::string &modelPath) {
  return engine_.loadModel(modelPath);
}

std::vector<Point2f> FaceLandmarkModel::getLandmarks(const ImageBuffer &input,
                                                     const FaceBox &face) {
  if (!engine_.getSession().GetInputCount())
    return {};

  // 1. Crop face
  cv::Mat img = input.getMat();

  // Pad the face box slightly to ensure landmarks are within context
  float pad = 0.0f;
  // PFLD is trained on tight crops usually, but some margin helps. Let's try
  // tight first.

  int x1 = std::max(0, (int)face.x1);
  int y1 = std::max(0, (int)face.y1);
  int x2 = std::min(img.cols, (int)face.x2);
  int y2 = std::min(img.rows, (int)face.y2);

  if (x2 <= x1 || y2 <= y1)
    return {};

  cv::Rect faceRect(x1, y1, x2 - x1, y2 - y1);
  cv::Mat faceImg = img(faceRect).clone();

  if (faceImg.empty())
    return {};

  // 2. Preprocess
  // PFLD input: 112x112
  // Normalization: (img / 255.0) -> [0, 1] ? Or specific mean/std?
  // Common PFLD implementation uses standard normalization or just /255.
  // We'll assume simple [0, 1] scaling or check if output looks weird.
  // PFLD often expects RGB.

  cv::Mat resized;
  cv::resize(faceImg, resized, cv::Size(inputSize_, inputSize_));
  cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

  cv::Mat floatImg;
  resized.convertTo(floatImg, CV_32F, 1.0 / 255.0); // 0-1

  // HWC -> CHW
  std::vector<int64_t> inputDims = {1, 3, inputSize_, inputSize_};
  size_t inputSize = 1 * 3 * inputSize_ * inputSize_;
  std::vector<float> inputData(inputSize);

  std::vector<cv::Mat> chans;
  cv::split(floatImg, chans);
  std::memcpy(inputData.data(), chans[0].data,
              chans[0].total() * sizeof(float));
  std::memcpy(inputData.data() + chans[0].total(), chans[1].data,
              chans[1].total() * sizeof(float));
  std::memcpy(inputData.data() + 2 * chans[0].total(), chans[2].data,
              chans[2].total() * sizeof(float));

  // 3. Inference
  std::vector<const char *> inputNames = {"input"};
  // PFLD input name check needed? Standard usually "input".
  // Try generic access if possible? No, ORT needs names.
  // Most PFLD onnx exports use 'input'.

  Ort::MemoryInfo memoryInfo =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<Ort::Value> inputTensors;
  inputTensors.push_back(Ort::Value::CreateTensor<float>(
      memoryInfo, inputData.data(), inputData.size(), inputDims.data(),
      inputDims.size()));

  std::vector<const char *> outputNames = {"output"};
  // PFLD output name often "output" or "landmark".

  auto outputs = engine_.run(inputNames, inputTensors, outputNames);
  if (outputs.empty())
    return {};

  // 4. Decode
  // Output: [1, 136] -> 68 points (x, y) normalized 0-1 (or -1 to 1?? PFLD
  // usually 0-1)
  float *landmarksPtr = outputs[0].GetTensorMutableData<float>();

  std::vector<Point2f> points;
  for (int i = 0; i < 68; ++i) {
    float lx = landmarksPtr[i * 2];
    float ly = landmarksPtr[i * 2 + 1];

    // Map back to original image
    Point2f p;
    p.x = x1 + lx * faceRect.width;
    p.y = y1 + ly * faceRect.height;
    points.push_back(p);
  }

  return points;
}

} // namespace AI
} // namespace PersonBeauty
