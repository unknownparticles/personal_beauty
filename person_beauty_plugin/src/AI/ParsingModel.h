#pragma once
#include "../Core/ImageBuffer.h"
#include "InferenceEngine.h"

namespace PersonBeauty {
namespace AI {

// Standard parsing classes (e.g. BiSeNet typically)
enum class ParsingClass {
  Background = 0,
  Skin = 1,
  LeftEyebrow = 2,
  RightEyebrow = 3,
  LeftEye = 4,
  RightEye = 5,
  Glasses = 6,
  LeftEar = 7,
  RightEar = 8,
  Earring = 9,
  Nose = 10,
  Mouth = 11,
  UpperLip = 12,
  LowerLip = 13,
  Neck = 14,
  Necklace = 15,
  Cloth = 16,
  Hair = 17,
  Hat = 18
};

class ParsingModel {
public:
  ParsingModel();
  bool load(const std::string &modelPath);

  // Process input image and return a mask where pixel values correspond to
  // ParsingClass
  std::shared_ptr<ImageBuffer> process(const ImageBuffer &input);

private:
  InferenceEngine engine_;
  int inputWidth_ = 512;
  int inputHeight_ = 512;
};

} // namespace AI
} // namespace PersonBeauty
