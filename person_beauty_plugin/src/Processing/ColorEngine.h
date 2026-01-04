#pragma once
#include "../Core/ImageBuffer.h"
#include <opencv2/opencv.hpp>

namespace PersonBeauty {
namespace Processing {

enum class BlendMode { Normal, Multiply, Screen, Overlay, SoftLight, Color };

class ColorEngine {
public:
  // Adjust brightness/contrast/saturation on masked region
  static void adjust(ImageBuffer &image, const ImageBuffer &mask,
                     float brightness, float contrast, float saturation,
                     float hue);

  // Blend two images using a mask
  static void blend(ImageBuffer &base, const ImageBuffer &blendLayer,
                    const ImageBuffer &mask, BlendMode mode, float opacity);

  // Neutral Gray Retouching: smooths skin while preserving texture
  static void applyNeutralGrayRetouch(ImageBuffer &image,
                                      const ImageBuffer &skinMask,
                                      float strength);

  // Neutral Gray Stereo: enhances facial features (Dodge & Burn)
  static void applyNeutralGrayStereo(ImageBuffer &image,
                                     const std::vector<cv::Point2f> &landmarks,
                                     float strength);
};

} // namespace Processing
} // namespace PersonBeauty
