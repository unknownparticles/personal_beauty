#include "ColorEngine.h"
#include <iostream>

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

  cv::Mat floatImg;
  image.getMat().convertTo(floatImg, CV_32F, 1.0 / 255.0);

  cv::Mat hsv;
  cv::cvtColor(floatImg, hsv, cv::COLOR_BGR2HSV);

  std::vector<cv::Mat> channels;
  cv::split(hsv, channels);

  cv::Mat floatMask;
  mask.getMat().convertTo(floatMask, CV_32F, 1.0 / 255.0);

  if (hue != 0) {
    channels[0] += hue;
  }

  if (saturation != 1.0f) {
    cv::Mat s_target = channels[1] * saturation;
    cv::Mat s_diff = s_target - channels[1];
    cv::Mat s_delta;
    cv::multiply(s_diff, floatMask, s_delta);
    channels[1] += s_delta;
  }

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
  floatImg.convertTo(image.getMat(), CV_8U, 255.0);
}

void ColorEngine::blend(ImageBuffer &base, const ImageBuffer &blendLayer,
                        const ImageBuffer &mask, BlendMode mode,
                        float opacity) {
  if (base.getMat().empty() || blendLayer.getMat().empty())
    return;

  cv::Mat baseF, blendF, maskF;
  base.getMat().convertTo(baseF, CV_32FC3, 1.0 / 255.0);
  blendLayer.getMat().convertTo(blendF, CV_32FC3, 1.0 / 255.0);

  if (!mask.getMat().empty()) {
    mask.getMat().convertTo(maskF, CV_32FC1, 1.0 / 255.0);
  } else {
    maskF = cv::Mat::ones(baseF.size(), CV_32FC1);
  }

  cv::Mat resultF = baseF.clone();

  // Optimized vectorized approach where possible
  if (mode == BlendMode::SoftLight) {
    // SoftLight: (1-2*L)*B^2 + 2*L*B
    // This is a simplified version, let's use the more accurate one:
    for (int y = 0; y < baseF.rows; ++y) {
      for (int x = 0; x < baseF.cols; ++x) {
        cv::Vec3f &b = baseF.at<cv::Vec3f>(y, x);
        cv::Vec3f &l = blendF.at<cv::Vec3f>(y, x);
        float m = maskF.at<float>(y, x) * opacity;
        cv::Vec3f res;
        for (int c = 0; c < 3; ++c) {
          if (l[c] < 0.5f) {
            res[c] = b[c] - (1.0f - 2.0f * l[c]) * b[c] * (1.0f - b[c]);
          } else {
            float d = (b[c] < 0.25f)
                          ? ((16.0f * b[c] - 12.0f) * b[c] + 4.0f) * b[c]
                          : std::sqrt(b[c]);
            res[c] = b[c] + (2.0f * l[c] - 1.0f) * (d - b[c]);
          }
          resultF.at<cv::Vec3f>(y, x)[c] = b[c] * (1.0f - m) + res[c] * m;
        }
      }
    }
  } else {
    // Fallback to normal blend for unimplemented modes
    cv::addWeighted(blendF, opacity, baseF, 1.0f - opacity, 0, resultF);
  }

  resultF.convertTo(base.getMat(), CV_8U, 255.0);
}

void ColorEngine::applyNeutralGrayRetouch(ImageBuffer &image,
                                          const ImageBuffer &skinMask,
                                          float strength) {
  cv::Mat img = image.getMat();
  cv::Mat floatImg;
  img.convertTo(floatImg, CV_32FC3, 1.0 / 255.0);

  cv::Mat blurred;
  cv::GaussianBlur(floatImg, blurred, cv::Size(21, 21), 0);

  // Gray Layer: 0.5 + (blurred - original) * strength
  cv::Mat grayLayer = cv::Mat::ones(img.size(), CV_32FC3) * 0.5f;
  grayLayer += (blurred - floatImg) * strength;

  ImageBuffer grayBuf(grayLayer);
  blend(image, grayBuf, skinMask, BlendMode::SoftLight, 1.0f);
}

void ColorEngine::applyNeutralGrayStereo(
    ImageBuffer &image, const std::vector<cv::Point2f> &landmarks,
    float strength) {
  if (landmarks.size() < 68)
    return;

  cv::Mat img = image.getMat();
  cv::Mat grayLayer = cv::Mat::ones(img.size(), CV_32FC3) * 0.5f;

  auto drawPattern = [&](const std::vector<int> &indices, float delta,
                         int blurSize) {
    if (indices.empty())
      return;
    std::vector<cv::Point> pts;
    for (int i : indices)
      pts.push_back(cv::Point((int)landmarks[i].x, (int)landmarks[i].y));

    cv::Mat featureMask = cv::Mat::zeros(img.size(), CV_32FC1);
    if (pts.size() > 2) {
      std::vector<std::vector<cv::Point>> contours = {pts};
      cv::fillPoly(featureMask, contours, cv::Scalar(1.0));
    } else if (pts.size() == 2) {
      cv::line(featureMask, pts[0], pts[1], cv::Scalar(1.0), 10);
    }

    cv::GaussianBlur(featureMask, featureMask,
                     cv::Size(blurSize * 2 + 1, blurSize * 2 + 1), 0);

    for (int c = 0; c < 3; ++c) {
      cv::Mat channel;
      cv::extractChannel(grayLayer, channel, c);
      channel += featureMask * delta * strength;
      cv::insertChannel(channel, grayLayer, c);
    }
  };

  // Dodge: T-Zone, Chin
  drawPattern({27, 28, 29, 30}, 0.2f, 20);      // Nose
  drawPattern({19, 20, 21, 22, 23}, 0.15f, 30); // Forehead
  drawPattern({8}, 0.1f, 20);                   // Chin

  // Burn: Face contour
  drawPattern({0, 1, 2, 3, 13, 14, 15, 16}, -0.2f, 40);

  ImageBuffer grayBuf(grayLayer);
  blend(image, grayBuf, ImageBuffer(), BlendMode::SoftLight, 1.0f);
}

} // namespace Processing
} // namespace PersonBeauty
