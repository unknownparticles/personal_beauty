#pragma once
#include <iostream>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

namespace PersonBeauty {
namespace AI {

class InferenceEngine {
public:
  InferenceEngine();
  ~InferenceEngine();

  bool loadModel(const std::string &modelPath);

  // Generic run method
  // In a real app, you might want more typed inputs/outputs or use template
  // methods
  std::vector<Ort::Value> run(const std::vector<const char *> &inputNames,
                              const std::vector<Ort::Value> &inputValues,
                              const std::vector<const char *> &outputNames);

  const Ort::Session &getSession() const { return *session_; }
  Ort::MemoryInfo &getMemoryInfo() { return memoryInfo_; }

private:
  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::Session> session_;
  Ort::MemoryInfo memoryInfo_ =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
};

} // namespace AI
} // namespace PersonBeauty
