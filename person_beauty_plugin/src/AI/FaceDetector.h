#pragma once
#include "../Core/ImageBuffer.h"
#include "InferenceEngine.h"
#include <vector>

namespace PersonBeauty {
namespace AI {

struct FaceBox {
  float x1, y1, x2, y2;
  float score;
};

class FaceDetector {
public:
  FaceDetector();
  bool load(const std::string &modelPath);

  // Returns list of detected faces
  std::vector<FaceBox> detect(const ImageBuffer &input);

private:
  void generateAnchors();
  void nonMaximumSuppression(std::vector<FaceBox> &input,
                             std::vector<FaceBox> &result, float iouThreshold);

  InferenceEngine engine_;
  // UltraFace-RFB-320 usually uses 320x240 input
  int inputWidth_ = 320;
  int inputHeight_ = 240;

  std::vector<std::vector<float>> anchors_;

  // Config for RFB-320
  const float center_variance_ = 0.1f;
  const float size_variance_ = 0.2f;
  const std::vector<std::vector<float>> min_boxes_ = {{10.0f, 16.0f, 24.0f},
                                                      {32.0f, 48.0f},
                                                      {64.0f, 96.0f},
                                                      {128.0f, 192.0f, 256.0f}};
  const std::vector<float> strides_ = {8.0f, 16.0f, 32.0f, 64.0f};
};

} // namespace AI
} // namespace PersonBeauty
