#include "FaceDetector.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <opencv2/opencv.hpp>

namespace PersonBeauty {
namespace AI {

FaceDetector::FaceDetector() { generateAnchors(); }

bool FaceDetector::load(const std::string &modelPath) {
  return engine_.loadModel(modelPath);
}

void FaceDetector::generateAnchors() {
  anchors_.clear();
  // Feature map sizes for 320x240:
  // 320/8=40, 240/8=30
  // 320/16=20, 240/16=15
  // 320/32=10, 240/32=8
  // Usually 240/32 is 7.5 -> 8 (ceil)

  // Let's calculate dynamically
  for (size_t i = 0; i < strides_.size(); ++i) {
    float stride = strides_[i];
    int feature_w = (int)std::ceil(inputWidth_ / stride);
    int feature_h = (int)std::ceil(inputHeight_ / stride);

    for (int y = 0; y < feature_h; ++y) {
      for (int x = 0; x < feature_w; ++x) {
        for (float min_size : min_boxes_[i]) {
          // cx, cy, w, h (normalized later?)
          // UltraFace anchors are normalized to [0,1]
          float cx = (x * stride + stride / 2.0f) / inputWidth_;
          float cy = (y * stride + stride / 2.0f) / inputHeight_;
          float w = min_size / inputWidth_;
          float h = min_size / inputHeight_;
          anchors_.push_back({cx, cy, w, h});
        }
      }
    }
  }
}

std::vector<FaceBox> FaceDetector::detect(const ImageBuffer &input) {
  if (!engine_.isLoaded())
    return {};

  // 1. Preprocess
  cv::Mat img = input.getMat();
  cv::Mat inputImg;
  cv::resize(img, inputImg, cv::Size(inputWidth_, inputHeight_));
  cv::cvtColor(inputImg, inputImg,
               cv::COLOR_BGR2RGB); // UltraFace likely expects RGB

  cv::Mat floatImg;
  inputImg.convertTo(floatImg, CV_32F);

  // (x - 127) / 128
  floatImg = (floatImg - 127.0f) / 128.0f;

  // HWC -> CHW
  // 1, 3, 240, 320
  std::vector<int64_t> inputDims = {1, 3, inputHeight_, inputWidth_};
  size_t inputSize = 1 * 3 * inputHeight_ * inputWidth_;
  std::vector<float> inputData(inputSize);

  std::vector<cv::Mat> chans;
  cv::split(floatImg, chans);
  // Copy to vector
  std::memcpy(inputData.data(), chans[0].data,
              chans[0].total() * sizeof(float));
  std::memcpy(inputData.data() + chans[0].total(), chans[1].data,
              chans[1].total() * sizeof(float));
  std::memcpy(inputData.data() + 2 * chans[0].total(), chans[2].data,
              chans[2].total() * sizeof(float));

  // 2. Inference
  std::vector<const char *> inputNames = {"input"};
  // Note: Some models name it "input", "input0", or "image".
  // Since we don't know for sure, let's verify names if possible, but
  // hardcoding "input" is standard for UltraFace.

  std::vector<Ort::Value> inputTensors;
  inputTensors.push_back(Ort::Value::CreateTensor<float>(
      engine_.getMemoryInfo(), inputData.data(), inputData.size(),
      inputDims.data(), inputDims.size()));

  std::vector<const char *> outputNames = {"scores", "boxes"};

  auto outputs = engine_.run(inputNames, inputTensors, outputNames);
  if (outputs.empty())
    return {};

  // 3. Post-process
  // scores: [1, N, 2]
  // boxes: [1, N, 4]

  float *scoresPtr = outputs[0].GetTensorMutableData<float>();
  float *boxesPtr = outputs[1].GetTensorMutableData<float>();

  // Get Shapes
  auto scoreShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
  int numAnchors = scoreShape[1];

  float confThreshold = 0.7f;
  std::vector<FaceBox> faces;

  for (int i = 0; i < numAnchors; ++i) {
    float score = scoresPtr[i * 2 + 1]; // Index 1 is face class usually
    if (score > confThreshold) {
      FaceBox face;
      // Decode box
      // boxesPtr usually: dx, dy, dw, dh
      float dx = boxesPtr[i * 4 + 0];
      float dy = boxesPtr[i * 4 + 1];
      float dw = boxesPtr[i * 4 + 2];
      float dh = boxesPtr[i * 4 + 3]; // Correct index is +3

      float cx = anchors_[i][0] + dx * center_variance_ * anchors_[i][2];
      float cy = anchors_[i][1] + dy * center_variance_ * anchors_[i][3];
      float w = anchors_[i][2] * std::exp(dw * size_variance_);
      float h = anchors_[i][3] * std::exp(dh * size_variance_);

      float x1 = (cx - w / 2.0f) * img.cols;
      float y1 = (cy - h / 2.0f) * img.rows;
      float x2 = (cx + w / 2.0f) * img.cols;
      float y2 = (cy + h / 2.0f) * img.rows;

      // Clip
      x1 = std::max(0.0f, x1);
      y1 = std::max(0.0f, y1);
      x2 = std::min((float)img.cols, x2);
      y2 = std::min((float)img.rows, y2);

      face.x1 = x1;
      face.y1 = y1;
      face.x2 = x2;
      face.y2 = y2;
      face.score = score;

      faces.push_back(face);
    }
  }

  std::vector<FaceBox> finalFaces;
  nonMaximumSuppression(faces, finalFaces, 0.3f);

  return finalFaces;
}

void FaceDetector::nonMaximumSuppression(std::vector<FaceBox> &input,
                                         std::vector<FaceBox> &result,
                                         float iouThreshold) {
  if (input.empty())
    return;

  // Sort by score
  std::sort(input.begin(), input.end(), [](const FaceBox &a, const FaceBox &b) {
    return a.score > b.score;
  });

  std::vector<bool> suppressed(input.size(), false);
  for (size_t i = 0; i < input.size(); ++i) {
    if (suppressed[i])
      continue;

    result.push_back(input[i]);
    float area1 = (input[i].x2 - input[i].x1) * (input[i].y2 - input[i].y1);

    for (size_t j = i + 1; j < input.size(); ++j) {
      if (suppressed[j])
        continue;

      float xx1 = std::max(input[i].x1, input[j].x1);
      float yy1 = std::max(input[i].y1, input[j].y1);
      float xx2 = std::min(input[i].x2, input[j].x2);
      float yy2 = std::min(input[i].y2, input[j].y2);

      float w = std::max(0.0f, xx2 - xx1);
      float h = std::max(0.0f, yy2 - yy1);
      float inter = w * h;

      float area2 = (input[j].x2 - input[j].x1) * (input[j].y2 - input[j].y1);
      float iou = inter / (area1 + area2 - inter);

      if (iou > iouThreshold) {
        suppressed[j] = true;
      }
    }
  }
}

} // namespace AI
} // namespace PersonBeauty
