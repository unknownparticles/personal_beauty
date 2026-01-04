#pragma once
#include "../Core/ImageBuffer.h"
#include <opencv2/opencv.hpp>
#include <vector>

namespace PersonBeauty {
namespace Processing {

struct LiquifyTool {
  float x, y;     // Center normalized 0-1
  float radius;   // Normalized 0-1
  float strength; // -1 to 1
};

class LiquifyEngine {
public:
  LiquifyEngine(int width, int height);

  // Reset mesh to identity
  void reset();

  // Apply a warp tool (Push/Puck/Bloat)
  // Note: Implementation specific (e.g. "Push" moves pixels away from center or
  // along direction)
  void push(float startX, float startY, float endX, float endY, float radius,
            float strength);
  void expand(float x, float y, float radius, float strength); // Bloat/Pinch

  // Apply the current mesh warp to an image
  // Returns warped image
  void process(const ImageBuffer &input, ImageBuffer &output);

private:
  int width_, height_;
  int meshRows_ = 32;
  int meshCols_ = 32;

  // Mesh vertices (absolute coordinates)
  std::vector<cv::Point2f> mesh_;

  void updateMaps();
  cv::Mat mapX_, mapY_;
  bool mapsDirty_ = true;
};

} // namespace Processing
} // namespace PersonBeauty
