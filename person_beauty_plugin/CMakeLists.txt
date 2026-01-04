cmake_minimum_required(VERSION 3.15)
project(PersonBeautyPlugin VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find CURL
find_package(CURL REQUIRED)

# Find ONNX Runtime (attempt to find via pkg-config or standard paths)
find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(ONNXRUNTIME onnxruntime)
endif()

if(NOT ONNXRUNTIME_FOUND)
    # Fallback to manual search in Homebrew locations
    find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
        PATHS /opt/homebrew/include/onnxruntime /usr/local/include/onnxruntime
    )
    find_library(ONNXRUNTIME_LIB onnxruntime
        PATHS /opt/homebrew/lib /usr/local/lib
    )
    if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIB)
        set(ONNXRUNTIME_FOUND TRUE)
        set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_INCLUDE_DIR})
        set(ONNXRUNTIME_LIBRARIES ${ONNXRUNTIME_LIB})
    endif()
endif()

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIRS}
    ${CURL_INCLUDE_DIRS}
)

# Core Library
add_library(PersonBeautyCore STATIC
    src/Core/ImageBuffer.h
    src/Core/Core.cpp
    src/AI/InferenceEngine.h
    src/AI/InferenceEngine.cpp
    src/AI/SegmentationModel.h
    src/AI/SegmentationModel.cpp
    src/AI/FaceDetector.h
    src/AI/FaceDetector.cpp
    src/AI/FaceLandmarkModel.h
    src/AI/FaceLandmarkModel.cpp
    src/AI/ParsingModel.h
    src/AI/ParsingModel.cpp
    src/Processing/MaskProcessor.h
    src/Processing/MaskProcessor.cpp
    src/Processing/ColorEngine.h
    src/Processing/ColorEngine.cpp
    src/Processing/LiquifyEngine.h
    src/Processing/LiquifyEngine.cpp
    src/Network/GenAPIClient.h
    src/Network/GenAPIClient.cpp
)

target_link_libraries(PersonBeautyCore PRIVATE
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBRARIES}
    ${CURL_LIBRARIES}
)

# Main Executable (for testing/development)
add_executable(PersonBeautyApp src/main.cpp)

target_link_libraries(PersonBeautyApp PRIVATE
    PersonBeautyCore
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBRARIES}
    ${CURL_LIBRARIES}
)
