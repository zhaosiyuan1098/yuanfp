#pragma once

#include "utils.h"


__global__ void convert_depth_to_xyz_map_kernel(
    const float *depth_on_device, int input_image_height,
    int input_image_width, float *xyz_map_on_device,
    float fx, float fy, float dx, float dy, float min_depth)
{
    const int row_idx = threadIdx.y + blockIdx.y * blockDim.y;
    const int col_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (row_idx >= input_image_height || col_idx >= input_image_width)
        return;

    const int pixel_idx = row_idx * input_image_width + col_idx;
    const float depth = depth_on_device[pixel_idx];
    if (depth < min_depth)
        return;

    const float x = (col_idx - dx) * depth / fx;
    const float y = (row_idx - dy) * depth / fy;
    const float z = depth;

    float *this_pixel_xyz = xyz_map_on_device + pixel_idx * 3;
    this_pixel_xyz[0] = x;
    this_pixel_xyz[1] = y;
    this_pixel_xyz[2] = z;
}

class Converter
{
public:
    Converter();
    ~Converter() = default;

    

    void convert_depth_to_xyz_map(const float *depth_on_device, int input_image_height,
                                  int input_image_width, float *xyz_map_on_device,
                                  float fx, float fy, float dx, float dy, float min_depth)
    {
        dim3 blockSize = {32, 32};
        dim3 gridSize = {ceil_div(input_image_width, 32), ceil_div(input_image_height, 32)};

        // 调用全局内核函数
        convert_depth_to_xyz_map_kernel<<<gridSize, blockSize, 0>>>(
            depth_on_device, input_image_height, input_image_width,
            xyz_map_on_device, fx, fy, dx, dy, min_depth);
    }
};
