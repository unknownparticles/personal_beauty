#include "GenAPIClient.h"
#include <curl/curl.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>

namespace PersonBeauty {
namespace Network {

GenAPIClient::GenAPIClient(const std::string &endpoint,
                           const std::string &apiKey)
    : endpoint_(endpoint), apiKey_(apiKey) {
  static bool curl_initialized = false;
  if (!curl_initialized) {
    curl_global_init(CURL_GLOBAL_ALL);
    curl_initialized = true;
  }
}

static std::string base64_encode(const cv::Mat &mat) {
  std::vector<uchar> buf;
  cv::imencode(".jpg", mat, buf);
  static const char *base64_chars =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string ret;
  int i = 0;
  unsigned char char_array_3[3];
  unsigned char char_array_4[4];

  for (auto const &c : buf) {
    char_array_3[i++] = c;
    if (i == 3) {
      char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
      char_array_4[1] =
          ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
      char_array_4[2] =
          ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
      char_array_4[3] = char_array_3[2] & 0x3f;
      for (i = 0; (i < 4); i++)
        ret += base64_chars[char_array_4[i]];
      i = 0;
    }
  }

  if (i) {
    int j;
    for (j = i; j < 3; j++)
      char_array_3[j] = '\0';
    char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
    char_array_4[1] =
        ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
    char_array_4[2] =
        ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
    char_array_4[3] = char_array_3[2] & 0x3f;
    for (j = 0; (j < i + 1); j++)
      ret += base64_chars[char_array_4[j]];
    while ((i++ < 3))
      ret += '=';
  }
  return ret;
}

void GenAPIClient::generate(const ImageBuffer &input, const ImageBuffer &mask,
                            const std::string &prompt, GenCallback callback) {

  std::thread([this, input, mask, prompt, callback]() {
    CURL *curl;
    CURLcode res;
    bool success = false;
    ImageBuffer resultImage(input.getMat().cols, input.getMat().rows, 3);

    curl = curl_easy_init();
    if (curl) {
      struct curl_slist *headers = NULL;
      headers = curl_slist_append(headers, "Content-Type: application/json");
      std::string auth = "Authorization: Bearer " + apiKey_;
      headers = curl_slist_append(headers, auth.c_str());

      std::string imgBase64 = base64_encode(input.getMat());
      std::string maskBase64 = base64_encode(mask.getMat());

      // Simple manual JSON construct
      std::string jsonPayload = "{\"prompt\": \"" + prompt +
                                "\", \"image\": \"" + imgBase64 +
                                "\", \"mask\": \"" + maskBase64 + "\"}";

      curl_easy_setopt(curl, CURLOPT_URL, endpoint_.c_str());
      curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
      curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonPayload.c_str());
      curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);

      std::cout << "[Network] Sending request to " << endpoint_ << " ..."
                << std::endl;

      res = curl_easy_perform(curl);

      if (res == CURLE_OK) {
        long response_code;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
        if (response_code == 200) {
          success = true;
          input.getMat().copyTo(resultImage.getMat());
        } else {
          std::cerr << "[Network] API returned error: " << response_code
                    << std::endl;
        }
      } else {
        std::cerr << "[Network] curl_easy_perform() failed: "
                  << curl_easy_strerror(res) << std::endl;
      }

      curl_easy_cleanup(curl);
      curl_slist_free_all(headers);
    }

    if (callback) {
      callback(success, resultImage);
    }
  }).detach();
}

} // namespace Network
} // namespace PersonBeauty
