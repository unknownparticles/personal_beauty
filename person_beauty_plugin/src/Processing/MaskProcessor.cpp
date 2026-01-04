#include "MaskProcessor.h"

namespace PersonBeauty {
namespace Processing {

void MaskProcessor::feather(ImageBuffer &mask, int radius) {
  if (radius <= 0)
    return;
  // Ensure odd kernel size
  int ksize = radius * 2 + 1;
  cv::GaussianBlur(mask.getMat(), mask.getMat(), cv::Size(ksize, ksize), 0);
}

void MaskProcessor::expand(ImageBuffer &mask, int pixels) {
  if (pixels <= 0)
    return;
  cv::Mat element = cv::getStructuringElement(
      cv::MORPH_ELLIPSE, cv::Size(2 * pixels + 1, 2 * pixels + 1));
  cv::dilate(mask.getMat(), mask.getMat(), element);
}

void MaskProcessor::shrink(ImageBuffer &mask, int pixels) {
  if (pixels <= 0)
    return;
  cv::Mat element = cv::getStructuringElement(
      cv::MORPH_ELLIPSE, cv::Size(2 * pixels + 1, 2 * pixels + 1));
  cv::erode(mask.getMat(), mask.getMat(), element);
}

void MaskProcessor::add(ImageBuffer &target, const ImageBuffer &source) {
  cv::bitwise_or(target.getMat(), source.getMat(), target.getMat());
}

void MaskProcessor::subtract(ImageBuffer &target, const ImageBuffer &source) {
  // target = target - source  => target & ~source
  cv::Mat notSource;
  cv::bitwise_not(source.getMat(), notSource);
  cv::bitwise_and(target.getMat(), notSource, target.getMat());
}

} // namespace Processing
} // namespace PersonBeauty
