#include "InferenceEngine.h"
#include <filesystem>
#include <iostream>

namespace PersonBeauty {
namespace AI {

InferenceEngine::InferenceEngine() {
  // Initialize ONNX Runtime Environment
  env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING,
                                    "PersonBeautyPlugin");
}

InferenceEngine::~InferenceEngine() {
  // Session and Env will be automatically destroyed
}

bool InferenceEngine::loadModel(const std::string &modelPath) {
  session_.reset();

  if (!std::filesystem::exists(modelPath)) {
    std::cerr << "[Warning] Model file not found: " << modelPath << std::endl;
    // Don't fail here if we want to allow the test to continue with mock data?
    // But loadModel usually returns success/fail.
    // The previous code returned false, so we stick to that.
    // The main.cpp handles failure by printing a warning.
    return false;
  }

  try {
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_BASIC);

    session_ = std::make_unique<Ort::Session>(*env_, modelPath.c_str(),
                                              sessionOptions);
    return true;
  } catch (const Ort::Exception &e) {
    std::cerr << "ONNX Runtime Error: " << e.what() << std::endl;
    session_.reset();
    return false;
  }
}

std::vector<Ort::Value>
InferenceEngine::run(const std::vector<const char *> &inputNames,
                     const std::vector<Ort::Value> &inputValues,
                     const std::vector<const char *> &outputNames) {
  if (!session_) {
    std::cerr << "Session not initialized!" << std::endl;
    return {};
  }

  try {
    return session_->Run(Ort::RunOptions{nullptr}, inputNames.data(),
                         inputValues.data(), inputValues.size(),
                         outputNames.data(), outputNames.size());
  } catch (const Ort::Exception &e) {
    std::cerr << "Inference Error: " << e.what() << std::endl;
    return {};
  }
}

} // namespace AI
} // namespace PersonBeauty
