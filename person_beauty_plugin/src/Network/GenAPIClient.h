#pragma once
#include "../Core/ImageBuffer.h"
#include <functional>
#include <future>
#include <string>
#include <vector>

namespace PersonBeauty {
namespace Network {

// Callback for async results
using GenCallback =
    std::function<void(bool success, const ImageBuffer &result)>;

class GenAPIClient {
public:
  GenAPIClient(const std::string &endpoint, const std::string &apiKey);

  // Send mask and prompt to external API for inpainting/generation
  // Returns a future that resolves when request is sent (or completes,
  // depending on design) Here we simulate async with std::async
  void generate(const ImageBuffer &input, const ImageBuffer &mask,
                const std::string &prompt, GenCallback callback);

private:
  std::string endpoint_;
  std::string apiKey_;
};

} // namespace Network
} // namespace PersonBeauty
