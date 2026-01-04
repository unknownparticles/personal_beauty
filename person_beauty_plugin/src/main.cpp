#include <chrono>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <vector>

#include "AI/FaceDetector.h"
#include "AI/FaceLandmarkModel.h"
#include "AI/ParsingModel.h"
#include "Core/ImageBuffer.h"
#include "Network/GenAPIClient.h"
#include "Processing/ColorEngine.h"
#include "Processing/LiquifyEngine.h"
#include "Processing/MaskProcessor.h"

using namespace PersonBeauty;

int main() {
  std::cout << "=== 人像美颜插件集成测试 (关键点优化 & 中性灰版) ==="
            << std::endl;

  // 1. 初始化与图像加载
  std::string imagePath = "../../test.jpg";
  std::cout << "[1/7] 加载测试图像: " << imagePath << " ..." << std::endl;

  cv::Mat rawImg = cv::imread(imagePath);
  if (rawImg.empty()) {
    std::cerr << "[错误] 无法加载图像: " << imagePath << std::endl;
    return 1;
  }

  int width = rawImg.cols;
  int height = rawImg.rows;
  ImageBuffer mainImage(width, height, 3);
  rawImg.copyTo(mainImage.getMat());

  // 2. AI 模型推理
  std::cout << "[2/7] 加载 AI 模型并运行推理..." << std::endl;

  AI::FaceDetector faceDetector;
  AI::ParsingModel parsingModel;
  AI::FaceLandmarkModel landmarkModel;

  std::string modelDir = "../models/";

  // Face Detector
  std::vector<AI::FaceBox> faces;
  if (faceDetector.load(modelDir + "face_detector.onnx")) {
    faces = faceDetector.detect(mainImage);
    std::cout << "      [人脸检测] 检测到 " << faces.size() << " 张人脸。"
              << std::endl;
  }

  // Face Landmarks & Debug visualization
  cv::Mat landmarkDebugImg = rawImg.clone();
  if (landmarkModel.load(modelDir + "face_landmark.onnx")) {
    std::cout << "      [关键点检测] 提取人脸关键点..." << std::endl;
    for (const auto &face : faces) {
      auto points = landmarkModel.getLandmarks(mainImage, face);
      if (!points.empty()) {
        for (const auto &p : points) {
          cv::circle(landmarkDebugImg, cv::Point((int)p.x, (int)p.y), 2,
                     cv::Scalar(0, 0, 255), -1); // RGB -> BGR(Red)
        }
      }
    }
    cv::imwrite("../../test_landmarks_debug.jpg", landmarkDebugImg);
    std::cout << "      [调试] 可视化已保存至 test_landmarks_debug.jpg"
              << std::endl;
  }

  // Skin Mask
  ImageBuffer skinMask(width, height, 1);
  skinMask.getMat() = cv::Scalar(0);
  if (parsingModel.load(modelDir + "face_parsing.onnx")) {
    auto parsingResult = parsingModel.process(mainImage);
    if (parsingResult) {
      const cv::Mat &segMap = parsingResult->getMat();
      cv::Mat segResized;
      cv::resize(segMap, segResized, cv::Size(width, height), 0, 0,
                 cv::INTER_NEAREST);
      cv::Mat skinMaskBinary =
          (segResized == 1) | (segResized == 10) | (segResized == 14);
      skinMaskBinary.convertTo(skinMask.getMat(), CV_8UC1, 255.0);
      Processing::MaskProcessor::feather(skinMask, 10);
      std::cout << "      [语义分割] 皮肤蒙版生成完毕。" << std::endl;
    }
  }

  // 3. 中性灰磨皮
  std::cout << "[3/7] 执行色彩调整与中性灰磨皮..." << std::endl;
  Processing::ColorEngine::adjust(mainImage, skinMask, 0.1f, 1.05f, 1.0f, 0.0f);
  Processing::ColorEngine::applyNeutralGrayRetouch(mainImage, skinMask, 0.7f);

  // 4. 自动瘦脸与中性灰立体增强
  std::cout << "[4/7] 执行关键点驱动特性 (瘦脸 & 立体型)..." << std::endl;
  Processing::LiquifyEngine liquify(width, height);
  for (const auto &face : faces) {
    auto pts = landmarkModel.getLandmarks(mainImage, face);
    if (!pts.empty()) {
      // 瘦脸
      liquify.slimFace(pts, 0.45f);
      // 立体增强
      Processing::ColorEngine::applyNeutralGrayStereo(mainImage, pts, 0.7f);
    }
  }
  liquify.process(mainImage, mainImage);

  // 5. 保存结果
  cv::imwrite("../../test_output.jpg", mainImage.getMat());
  std::cout << "[5/7] 结果已保存至 test_output.jpg" << std::endl;

  // 6. 网络调用测试
  std::cout << "[6/7] 调用外部 API (模拟)..." << std::endl;
  Network::GenAPIClient genClient("https://api.example.com/generate", "key");
  genClient.generate(
      mainImage, skinMask, "Vogue style", [](bool s, const ImageBuffer &r) {
        std::cout << "      [API 回调] " << (s ? "成功" : "失败 (模拟环境)")
                  << std::endl;
      });

  std::this_thread::sleep_for(std::chrono::seconds(1));
  std::cout << "=== 集成测试完成 ===" << std::endl;
  return 0;
}
