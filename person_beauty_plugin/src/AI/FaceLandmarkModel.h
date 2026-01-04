#pragma once
#include "../Core/ImageBuffer.h"
#include "FaceDetector.h" // for FaceBox
#include "InferenceEngine.h"

namespace PersonBeauty {
namespace AI {

struct Point2f {
  float x, y;
};

class FaceLandmarkModel {
public:
  FaceLandmarkModel();
  bool load(const std::string &modelPath);

  // Process input image and a face box to get landmarks
  // Returns 68 or 106 points depending on model
  std::vector<Point2f> getLandmarks(const ImageBuffer &input,
                                    const FaceBox &face);

private:
  InferenceEngine engine_;
  int inputSize_ = 112; // Typical for landmark models
};

} // namespace AI
} // namespace PersonBeauty
