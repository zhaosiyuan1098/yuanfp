#pragma once

#include <memory>
#include <iostream>

#include <eigen3/Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <cuda_runtime.h>
#include "cuda.h"
#include <gtest/gtest.h>
#include <glog/logging.h>

static const std::string refiner_engine_path_ = "/workspace/models/refiner_hwc_dynamic_fp16.engine";
static const std::string scorer_engine_path_ = "/workspace/models/scorer_hwc_dynamic_fp16.engine";
static const std::string demo_data_path_ = "/workspace/test_data/mustard0";
static const std::string demo_textured_obj_path = demo_data_path_ + "/mesh/textured_simple.obj";
static const std::string demo_textured_map_path = demo_data_path_ + "/mesh/texture_map.png";
static const std::string demo_name_ = "mustard";
static const std::string frame_id = "1581120424100262102";

const std::string first_rgb_path = demo_data_path_ + "/rgb/" + frame_id + ".png";
const std::string first_depth_path = demo_data_path_ + "/depth/" + frame_id + ".png";
const std::string first_mask_path = demo_data_path_ + "/masks/" + frame_id + ".png";

#define MAX_INPUT_IMAGE_HEIGHT 1080
#define MAX_INPUT_IMAGE_WIDTH 1920


#define CHECK_STATE(state, hint) \
  {                              \
    if (!(state))                \
    {                            \
      LOG(ERROR) << (hint);      \
      return false;              \
    }                            \
  }

#define MESSURE_DURATION(run)                                                                \
  {                                                                                          \
    auto start = std::chrono::high_resolution_clock::now();                                  \
    (run);                                                                                   \
    auto end = std::chrono::high_resolution_clock::now();                                    \
    LOG(INFO) << #run << " cost(us): "                                                       \
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count(); \
  }

#define MESSURE_DURATION_AND_CHECK_STATE(run, hint)                                          \
  {                                                                                          \
    auto start = std::chrono::high_resolution_clock::now();                                  \
    CHECK_STATE((run), hint);                                                                \
    auto end = std::chrono::high_resolution_clock::now();                                    \
    LOG(INFO) << #run << " cost(us): "                                                       \
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count(); \
  }


#define CHECK_CUDA(result, hint) \
{ \
  auto res = (result); \
  if (res != cudaSuccess) { \
    LOG(ERROR) << hint << "  CudaError: " << res; \
    return false; \
  } \
}


static uint16_t ceil_div(uint16_t numerator, uint16_t denominator)
    {
        return (numerator + denominator - 1) / denominator;
    }
