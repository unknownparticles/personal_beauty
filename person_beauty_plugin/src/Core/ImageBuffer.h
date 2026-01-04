#pragma once
#include <memory>
#include <opencv2/opencv.hpp>

namespace PersonBeauty {

class ImageBuffer {
public:
  ImageBuffer() {}
  ImageBuffer(const cv::Mat &mat) : data(mat) {}
  ImageBuffer(int width, int height, int channels = 3) {
    int type = CV_8UC3;
    if (channels == 1)
      type = CV_8UC1;
    else if (channels == 4)
      type = CV_8UC4;
    data = cv::Mat(height, width, type);
  }

  cv::Mat &getMat() { return data; }
  const cv::Mat &getMat() const { return data; }

private:
  cv::Mat data;
};

} // namespace PersonBeauty
