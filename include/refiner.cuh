#pragma once

#include "utils.h"
#include "checker.h"
#include "dataKeeper.h"

__global__ void erode_depth_kernel(
    float *depth, float *out, int H, int W, int radius, float depth_diff_thres, float ratio_thres,
    float zfar)
{
  int h = blockIdx.y * blockDim.y + threadIdx.y;
  int w = blockIdx.x * blockDim.x + threadIdx.x;

  if (w >= W || h >= H)
  {
    return;
  }

  float d_ori = depth[h * W + w];

  // Check the validity of the depth value
  if (d_ori < 0.1f || d_ori >= zfar)
  {
    out[h * W + w] = 0.0f;
    return;
  }

  float bad_cnt = 0.0f;
  float total = 0.0f;

  // Loop over the neighboring pixels
  for (int u = w - radius; u <= w + radius; u++)
  {
    if (u < 0 || u >= W)
    {
      continue;
    }
    for (int v = h - radius; v <= h + radius; v++)
    {
      if (v < 0 || v >= H)
      {
        continue;
      }
      float cur_depth = depth[v * W + u];

      total += 1.0f;

      if (cur_depth < 0.1f || cur_depth >= zfar || fabsf(cur_depth - d_ori) > depth_diff_thres)
      {
        bad_cnt += 1.0f;
      }
    }
  }

  // Check the ratio of bad pixels
  if ((bad_cnt / total) > ratio_thres)
  {
    out[h * W + w] = 0.0f;
  }
  else
  {
    out[h * W + w] = d_ori;
  }
}

class Refiner
{
private:
  std::vector<Eigen::Matrix4f> pre_compute_rotations_;
  cudaStream_t cuda_stream_;
  float *erode_depth_buffer_device_;
  // Add other necessary member variables

public:
  Refiner(std::string engine_path)
  {
    Checker checker;
    if (!checker.check_file_suffix(engine_path, ".engine"))
    {
      throw std::invalid_argument("Invalid engine path");
    };
    // Initialize member variables as needed
    std::cout << "Refiner created" << std::endl;
  };

  ~Refiner() {
    // Cleanup if necessary
  };

  void erode_depth(
      cudaStream_t stream, float *depth, float *out, int H, int W, int radius=2,
      float depth_diff_thres=0.001, float ratio_thres=0.8, float zfar=100)
  {
    dim3 block(16, 16);
    dim3 grid(ceil_div(W, 16), ceil_div(H, 16), 1);

    erode_depth_kernel<<<grid, block, 0, stream>>>(
        depth, out, H, W, radius, depth_diff_thres, ratio_thres, zfar);
  }

  bool GetHypPoses(void *_depth_on_device,
                   void *_mask_on_host,
                   int input_image_height,
                   int input_image_width,
                   std::vector<Eigen::Matrix4f> & out_hyp_poses)
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
    erode_depth(cuda_stream_, depth_on_device, erode_depth_buffer_device_,
                input_image_height, input_image_width); // Example parameters

    return true; // Or appropriate return value
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
