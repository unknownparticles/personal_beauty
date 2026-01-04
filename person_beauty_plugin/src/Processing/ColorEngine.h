#pragma once
#include <opencv2/opencv.hpp>
#include "../Core/ImageBuffer.h"

namespace PersonBeauty {
namespace Processing {

enum class BlendMode {
    Normal,
    Multiply,
    Screen,
    Overlay,
    SoftLight,
    Color
};

class ColorEngine {
public:
    // Adjust brightness/contrast/saturation on masked region
    static void adjust(ImageBuffer& image, const ImageBuffer& mask, 
                       float brightness, float contrast, float saturation, float hue);

    // Blend two images using a mask
    static void blend(ImageBuffer& base, const ImageBuffer& blendLayer, const ImageBuffer& mask, BlendMode mode, float opacity);
};

}
}
