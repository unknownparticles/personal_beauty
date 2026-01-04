#pragma once
#include "../Core/ImageBuffer.h"
#include "InferenceEngine.h"

namespace PersonBeauty {
namespace AI {

class SegmentationModel {
public:
  SegmentationModel();
  ~SegmentationModel() = default;

  bool load(const std::string &modelPath);

  // Process input image and return memory buffer with mask
  // Input is assumed to be BGR (OpenCV default)
  // Output is single channel 0-255 mask
  std::shared_ptr<ImageBuffer> process(const ImageBuffer &input);

private:
  InferenceEngine engine_;
  int inputWidth_ = 256; // Default, should be read from model
  int inputHeight_ = 256;
};

} // namespace AI
} // namespace PersonBeauty
