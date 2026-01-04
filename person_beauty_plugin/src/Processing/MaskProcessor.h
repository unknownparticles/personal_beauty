#pragma once
#include "../Core/ImageBuffer.h"
#include <opencv2/opencv.hpp>

namespace PersonBeauty {
namespace Processing {

class MaskProcessor {
public:
  static void feather(ImageBuffer &mask, int radius);
  static void expand(ImageBuffer &mask, int pixels);
  static void shrink(ImageBuffer &mask, int pixels);

  // Combine multiple masks (e.g. Skin + Neck)
  static void add(ImageBuffer &target, const ImageBuffer &source);
  static void subtract(ImageBuffer &target, const ImageBuffer &source);
};

} // namespace Processing
} // namespace PersonBeauty
