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
  std::cout << "=== 人像美颜插件集成测试 (全功能版) ===" << std::endl;

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
  std::cout << "      图像加载成功: " << width << "x" << height << std::endl;

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

  // landmarks visualization img
  cv::Mat landmarkDebugImg = rawImg.clone();

  // Face Landmarks
  if (landmarkModel.load(modelDir + "face_landmark.onnx")) {
    std::cout << "      [关键点检测] 提取人脸关键点..." << std::endl;
    for (const auto &face : faces) {
      auto points = landmarkModel.getLandmarks(mainImage, face);
      std::cout << "          已提取 " << points.size() << " 个点。"
                << std::endl;
      // Draw
      for (const auto &p : points) {
        cv::circle(landmarkDebugImg, cv::Point((int)p.x, (int)p.y), 2,
                   cv::Scalar(0, 255, 0), -1);
      }
    }
    cv::imwrite("../../test_landmarks_debug.jpg", landmarkDebugImg);
    std::cout
        << "      [调试] 关键点可视化已保存至 ../../test_landmarks_debug.jpg"
        << std::endl;
  }

  ImageBuffer skinMask(width, height, 1);
  skinMask.getMat() = cv::Scalar(0);

  // Parsing Model
  if (parsingModel.load(modelDir + "face_parsing.onnx")) {
    std::cout << "      [语义分割] 正在生成人脸区域蒙版..." << std::endl;
    auto parsingResult = parsingModel.process(mainImage);
    if (!parsingResult) {
      std::cerr << "      [警告] 语义分割结果为空，跳过皮肤蒙版生成。"
                << std::endl;
    }
    if (parsingResult) {
      const cv::Mat &segMap = parsingResult->getMat();
      cv::Mat segResized;
      if (segMap.size() == skinMask.getMat().size()) {
        segResized = segMap;
      } else {
        cv::resize(segMap, segResized, skinMask.getMat().size(), 0, 0,
                   cv::INTER_NEAREST);
      }

      // 使用向量化操作提取皮肤区域 (label: 1/10/14)，避免手写双重循环
      cv::Mat skinMaskBinary =
          (segResized == 1) | (segResized == 10) | (segResized == 14);
      skinMaskBinary.convertTo(skinMask.getMat(), CV_8UC1, 255.0);
      std::cout << "      已生成皮肤蒙版。" << std::endl;
      cv::imwrite("../../test_mask_debug.png", skinMask.getMat());
    }
  }

  // 3. 蒙版处理
  std::cout << "[3/7] 处理蒙版（羽化）..." << std::endl;
  Processing::MaskProcessor::feather(skinMask, 5);

  // 4. 色彩调整
  std::cout << "[4/7] 执行色彩调整（美白/提亮）..." << std::endl;
  Processing::ColorEngine::adjust(mainImage, skinMask, 0.15f, 1.1f, 0.9f, 0.0f);

  // 5. 液化变形
  std::cout << "[5/7] 执行液化变形 (自动瘦脸)..." << std::endl;
  Processing::LiquifyEngine liquify(width, height);
  if (!faces.empty()) {
    // 针对检测到的每一张人脸进行瘦脸
    for (const auto &face : faces) {
      auto pts = landmarkModel.getLandmarks(mainImage, face);
      if (!pts.empty()) {
        liquify.slimFace(pts, 0.45f); // 45% 瘦脸强度
      }
    }
    std::cout << "      已应用自动瘦脸变形。" << std::endl;
  }
  liquify.process(mainImage, mainImage);

  // 6. 保存结果
  std::string outPath = "../../test_output.jpg";
  std::cout << "[6/7] 保存结果至: " << outPath << std::endl;
  cv::imwrite(outPath, mainImage.getMat());

  // 7. 外部 API 调用 (HTTP 调用测试)
  std::cout << "[7/7] 调用外部生成 API (网络测试)..." << std::endl;
  Network::GenAPIClient genClient("https://api.example.com/generate",
                                  "dummy_key");
  bool apiCalled = false;
  genClient.generate(mainImage, skinMask, "Make skin smoother",
                     [&apiCalled](bool success, const ImageBuffer &result) {
                       if (success) {
                         std::cout << "      [网络回调] API 调用成功！"
                                   << std::endl;
                       } else {
                         std::cout << "      [网络回调] API 调用失败 "
                                      "(符合预期，由于使用了模拟端点)。"
                                   << std::endl;
                       }
                       apiCalled = true;
                     });

  // Wait for async response
  for (int i = 0; i < 10 && !apiCalled; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

  std::cout << "=== 集成测试完成 ===" << std::endl;
  return 0;
}
