#pragma once

#include "utils.h"
#include "checker.h"
#include "dataKeeper.h"
#include "eroder.cuh"
#include "bilateral_filter.cuh"




class Refiner
{
private:
  std::vector<Eigen::Matrix4f> pre_compute_rotations_;
  cudaStream_t cuda_stream_;
  const float min_depth_=10;
  const Eigen::Matrix3f intrinsic_matrix;
  std::shared_ptr<Eroder> eroder_;
  std::shared_ptr<Bilateral_Filter> bilateral_filter_;

public:
  Refiner(std::string engine_path)
  {
    Checker checker;
    if (!checker.check_file_suffix(engine_path, ".engine"))
    {
      throw std::invalid_argument("Invalid engine path");
    };
    // 初始化成员变量
    cudaStreamCreate(&cuda_stream_);
    cudaMalloc(&eroder_->erode_depth_buffer_device_, sizeof(float) * MAX_INPUT_IMAGE_HEIGHT * MAX_INPUT_IMAGE_WIDTH);
    cudaMalloc(&bilateral_filter_->bilateral_depth_buffer_device_, sizeof(float) * MAX_INPUT_IMAGE_HEIGHT * MAX_INPUT_IMAGE_WIDTH);
    std::cout << "Refiner created" << std::endl;
  };

  ~Refiner()
  {
    // 释放设备内存
    cudaFree(eroder_->erode_depth_buffer_device_);
    cudaFree(bilateral_filter_->bilateral_depth_buffer_device_);
    cudaStreamDestroy(cuda_stream_);
  };


  bool GuessTranslation(
      const Eigen::MatrixXf &depth, const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &mask, const Eigen::Matrix3f &K,
      float min_depth, Eigen::Vector3f &center)
  {
    // Find the indices where mask is positive
    std::vector<int> vs, us;
    for (int i = 0; i < mask.rows(); i++)
    {
      for (int j = 0; j < mask.cols(); j++)
      {
        if (mask(i, j) > 0)
        {
          vs.push_back(i);
          us.push_back(j);
        }
      }
    }
    CHECK_STATE(!us.empty(), "[FoundationposeSampling] Mask is all zero.");

    float uc =
        (*std::min_element(us.begin(), us.end()) + *std::max_element(us.begin(), us.end())) / 2.0;
    float vc =
        (*std::min_element(vs.begin(), vs.end()) + *std::max_element(vs.begin(), vs.end())) / 2.0;

    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> valid =
        (mask.array() > 0) && (depth.array() >= min_depth);
    CHECK_STATE(valid.any(), "[FoundationposeSampling] No valid value in mask.");

    std::vector<float> valid_depth;
    for (int i = 0; i < valid.rows(); i++)
    {
      for (int j = 0; j < valid.cols(); j++)
      {
        if (valid(i, j))
        {
          valid_depth.push_back(depth(i, j));
        }
      }
    }
    std::sort(valid_depth.begin(), valid_depth.end());
    int n = valid_depth.size();
    float zc =
        (n % 2 == 0) ? (valid_depth[n / 2 - 1] + valid_depth[n / 2]) / 2.0 : valid_depth[n / 2];

    center = K.inverse() * Eigen::Vector3f(uc, vc, 1) * zc;
    return true;
  }

  bool GetHypPoses(void *_depth_on_device,
                   void *_mask_on_host,
                   int input_image_height,
                   int input_image_width,
                   std::vector<Eigen::Matrix4f> &out_hyp_poses)
  {
    if (_depth_on_device == nullptr || _mask_on_host == nullptr)
    {
      throw std::invalid_argument("[Refiner:GetHypPoses] Got INVALID depth/mask ptr on device!!!");
    }

    // 1. Generate initial hypothesis poses
    out_hyp_poses = pre_compute_rotations_;

    // 2. Optimize depth map
    float *depth_on_device = static_cast<float *>(_depth_on_device);
    // Assuming radius, depth_diff_thres, ratio_thres, zfar are member variables or defined elsewhere
    eroder_->erode_depth(cuda_stream_, depth_on_device, eroder_->erode_depth_buffer_device_,
                input_image_height, input_image_width); // Example parameters

    bilateral_filter_->bilateral_filter_depth(cuda_stream_,
                           eroder_->erode_depth_buffer_device_,
                           bilateral_filter_->bilateral_depth_buffer_device_,
                           input_image_height,
                           input_image_width);

    return true; // Or appropriate return value

    // 2.3 拷贝到host端缓存
    cudaMemcpyAsync(bilateral_filter_->bilateral_depth_buffer_host_.data(),
                    bilateral_filter_->bilateral_depth_buffer_device_,
                    input_image_height * input_image_width * sizeof(float),
                    cudaMemcpyDeviceToHost,
                    cuda_stream_);

    // 2.4 同步cuda流
    CHECK_CUDA(cudaStreamSynchronize(cuda_stream_),
               "[FoundationPoseSampling] cudaStreamSync `cuda_stream_` FAILED!!!");

    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        bilateral_filter_depth_host(bilateral_filter_->bilateral_depth_buffer_host_.data(),
                                    input_image_height,
                                    input_image_width);
    Eigen::Map<Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        mask_host(static_cast<uint8_t *>(_mask_on_host),
                  input_image_height,
                  input_image_width);

    Eigen::Vector3f center;
    CHECK_STATE(GuessTranslation(bilateral_filter_depth_host,
                                 mask_host,
                                 intrinsic_matrix,
                                 min_depth_,
                                 center),
                "[FoundationPose Sampling] Failed to GuessTranslation!!!");

    LOG(INFO) << "[FoundationPose Sampling] Center: " << center;

    // 4. 把三维中心放到变换矩阵内
    for (auto &pose : out_hyp_poses)
    {
      pose.block<3, 1>(0, 3) = center;
    }

    return true;
  }

  void refinePreProcess(std::shared_ptr<DataKeeper> datakeeper)
  {
    auto local_datakeeper = std::dynamic_pointer_cast<DataKeeper>(datakeeper);
    if (!local_datakeeper)
    {
      throw std::invalid_argument("[Refiner:RefinePreProcess] Invalid DataKeeper");
    }
    if (local_datakeeper->rgb_on_host.empty() || local_datakeeper->depth_on_host.empty())
    {
      throw std::invalid_argument("Empty RGB or Depth");
    }

    std::cout << "RefinePreProcess" << std::endl;
  }
};