#include "ColorEngine.h"

namespace PersonBeauty {
namespace Processing {

void ColorEngine::adjust(ImageBuffer &image, const ImageBuffer &mask,
                         float brightness, float contrast, float saturation,
                         float hue) {
  if (image.getMat().size() != mask.getMat().size()) {
    std::cerr << "[Error] ColorEngine: Image and Mask size mismatch!"
              << std::endl;
    return;
  }

  // 1. Convert to float for processing
  cv::Mat floatImg;
  image.getMat().convertTo(floatImg, CV_32F, 1.0 / 255.0);

  cv::Mat hsv;
  cv::cvtColor(floatImg, hsv, cv::COLOR_BGR2HSV);

  // Split channels
  std::vector<cv::Mat> channels;
  cv::split(hsv, channels); // H, S, V

  // Apply adjustments using mask
  // Note: Mask is 8UC1 (0-255). Normalize to 0-1 float for blending
  cv::Mat floatMask;
  mask.getMat().convertTo(floatMask, CV_32F, 1.0 / 255.0);

  // Hue
  if (hue != 0) {
    // H is 0-360 usually in float HSV? in OpenCV float it's 0-360
    channels[0] += hue;
    // Wrap around logic needed... for simplicity omitted
  }

  // Saturation
  if (saturation != 1.0f) {
    // Blending logic: S_new = S_old * saturation
    // But only where mask is 1. Better: S_target = S_old * saturation. Final =
    // S_old * (1-mask) + S_target * mask
    cv::Mat s_target = channels[1] * saturation;
    cv::Mat s_diff = s_target - channels[1];
    cv::Mat s_delta;
    cv::multiply(s_diff, floatMask, s_delta);
    channels[1] += s_delta;
  }

  // Value (Brightness/Contrast)
  // Contrast usually applied around mid-gray (0.5)
  if (contrast != 1.0f || brightness != 0.0f) {
    cv::Mat v_target;
    cv::subtract(channels[2], 0.5, v_target);
    cv::multiply(v_target, contrast, v_target);
    cv::add(v_target, 0.5 + brightness, v_target);

    cv::Mat v_diff = v_target - channels[2];
    cv::Mat v_delta;
    cv::multiply(v_diff, floatMask, v_delta);
    channels[2] += v_delta;
  }

  cv::merge(channels, hsv);
  cv::cvtColor(hsv, floatImg, cv::COLOR_HSV2BGR);

  // Convert back to 8U
  floatImg.convertTo(image.getMat(), CV_8U, 255.0);
}

void ColorEngine::blend(ImageBuffer &base, const ImageBuffer &blendLayer,
                        const ImageBuffer &mask, BlendMode mode,
                        float opacity) {
  // Simplified blend implementation
  // Ideally use float buffers

  // TODO: Implement blend modes
}

} // namespace Processing
} // namespace PersonBeauty
