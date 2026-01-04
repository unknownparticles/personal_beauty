#include "LiquifyEngine.h"

namespace PersonBeauty {
namespace Processing {

LiquifyEngine::LiquifyEngine(int width, int height)
    : width_(width), height_(height) {
  reset();
}

void LiquifyEngine::reset() {
  mesh_.clear();
  // Grid includes edges, so +1
  float cellW = (float)width_ / meshCols_;
  float cellH = (float)height_ / meshRows_;

  for (int y = 0; y <= meshRows_; ++y) {
    for (int x = 0; x <= meshCols_; ++x) {
      mesh_.push_back(cv::Point2f(x * cellW, y * cellH));
    }
  }
  mapsDirty_ = true;
}

void LiquifyEngine::push(float startX, float startY, float endX, float endY,
                         float radius, float strength) {
  // Convert normalized coords to absolute
  cv::Point2f start(startX * width_, startY * height_);
  cv::Point2f end(endX * width_, endY * height_);
  cv::Point2f vec = end - start;
  float r = radius * std::max(width_, height_);
  float r2 = r * r;

  // Simple deformation: move mesh points based on distance to segment or point
  // For simplicity, just moving points near 'start' towards 'vec'
  for (auto &pt : mesh_) {
    float dist2 = (pt.x - start.x) * (pt.x - start.x) +
                  (pt.y - start.y) * (pt.y - start.y);
    if (dist2 < r2) {
      float factor = (1.0f - dist2 / r2) * strength;
      pt += vec * factor;
    }
  }
  mapsDirty_ = true;
}

void LiquifyEngine::expand(float x, float y, float radius, float strength) {
  cv::Point2f center(x * width_, y * height_);
  float r = radius * std::max(width_, height_);
  float r2 = r * r;

  for (auto &pt : mesh_) {
    float dx = pt.x - center.x;
    float dy = pt.y - center.y;
    float dist2 = dx * dx + dy * dy;

    if (dist2 < r2 && dist2 > 0) {
      float dist = std::sqrt(dist2);
      float factor = (1.0f - dist2 / r2) * strength;

      // Move away (bloat) or Towards (pinch)
      // If strength > 0 bloat
      cv::Point2f dir(dx / dist, dy / dist);
      pt += dir * (factor * r * 0.1f);
    }
  }
  mapsDirty_ = true;
}

void LiquifyEngine::updateMaps() {
  if (!mapsDirty_)
    return;

  mapX_.create(height_, width_, CV_32FC1);
  mapY_.create(height_, width_, CV_32FC1);

  // Interpolate mesh to pixel maps
  // This is the heavy part. For 32x32 mesh on 4k img, need efficient
  // interpolation. Ideally use hardware acceleration or simpler bilinear from
  // mesh. For this demo, we'll try using cv::resize from a coarse map.

  // 1. Create coarse maps from mesh
  cv::Mat coarseX(meshRows_ + 1, meshCols_ + 1, CV_32FC1);
  cv::Mat coarseY(meshRows_ + 1, meshCols_ + 1, CV_32FC1);

  for (int y = 0; y <= meshRows_; ++y) {
    for (int x = 0; x <= meshCols_; ++x) {
      int idx = y * (meshCols_ + 1) + x;
      coarseX.at<float>(y, x) = mesh_[idx].x;
      coarseY.at<float>(y, x) = mesh_[idx].y;
    }
  }

  // 2. Resize to full resolution (Linear interpolation)
  cv::resize(coarseX, mapX_, cv::Size(width_, height_), 0, 0, cv::INTER_LINEAR);
  cv::resize(coarseY, mapY_, cv::Size(width_, height_), 0, 0, cv::INTER_LINEAR);

  mapsDirty_ = false;
}

void LiquifyEngine::process(const ImageBuffer &input, ImageBuffer &output) {
  updateMaps();
  // Remap
  cv::remap(input.getMat(), output.getMat(), mapX_, mapY_, cv::INTER_LINEAR);
}

} // namespace Processing
} // namespace PersonBeauty
