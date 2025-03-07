#pragma once

#include "utils.h"
#include "dataKeeper.h"
#include "converter.cuh"

class Loader
{
private:
    /* data */
public:
    Loader(/* args */) = default;
    ~Loader() = default;

    Eigen::Matrix3f load_camK()
    {
        std::cout << "Loading camera matrix" << std::endl;
        Eigen::Matrix3f camK;
        camK << 520.9, 0, 325.1,
            0, 521.0, 249.7,
            0, 0, 1;
        return camK;
    }

    std::tuple<cv::Mat, cv::Mat, cv::Mat> load_rgb_depth_mask(std::string rgb_path, std::string depth_path, std::string mask_path)
    {
        std::cout << "Loading RGB, Depth and Mask" << std::endl;
        cv::Mat rgb = cv::imread(rgb_path, cv::IMREAD_COLOR);
        cv::Mat depth = cv::imread(depth_path, cv::IMREAD_ANYDEPTH);
        cv::Mat mask = cv::imread(mask_path, cv::IMREAD_GRAYSCALE);
        return std::make_tuple(rgb, depth, mask);
    }

    std::tuple<cv::Mat, cv::Mat> load_rgb_depth(std::string rgb_path, std::string depth_path)
    {
        std::cout << "Loading RGB and Depth" << std::endl;
        cv::Mat rgb = cv::imread(rgb_path, cv::IMREAD_COLOR);
        cv::Mat depth = cv::imread(depth_path, cv::IMREAD_ANYDEPTH);
        return std::make_tuple(rgb, depth);
    }
    // std::vector<std::filesystem::path> load_files_in_dir(const std::filesystem::path &dir_path)
    // {
    //     std::cout << "Loading files in directory" << std::endl;
    // }

    bool upload_to_device(const cv::Mat &rgb,
                          const cv::Mat &depth,
                          const cv::Mat &mask,
                          const Eigen::Matrix3f &intrinsic_matrix,
                          std::shared_ptr<DataKeeper> dataKeeper)
    {
        const size_t input_image_height = rgb.rows;
        const size_t input_image_width = rgb.cols;
        const size_t input_image_pixel_num = input_image_height * input_image_width;
        void *rgb_on_device = nullptr,
             *depth_on_device = nullptr,
             *xyz_map_on_device = nullptr;

        CHECK_CUDA(cudaMalloc(&rgb_on_device,
                              input_image_pixel_num * 3 * sizeof(uint8_t)),
                   "[Loader:upload_to_device] RefinePreProcess malloc managed `rgb_on_device` failed");
        CHECK_CUDA(cudaMemcpy(rgb_on_device,
                              dataKeeper->rgb_on_host.data,
                              input_image_pixel_num * 3 * sizeof(uint8_t),
                              cudaMemcpyHostToDevice),
                   "[Loader:upload_to_device] cudaMemcpy rgb_host -> rgb_device failed");

        CHECK_CUDA(cudaMalloc(&depth_on_device,
                              input_image_pixel_num * sizeof(float)),
                   "[Loader:upload_to_device] RefinePreProcess malloc managed `depth_on_device` failed!!!");
        CHECK_CUDA(cudaMemcpy(depth_on_device,
                              dataKeeper->depth_on_host.data,
                              input_image_pixel_num * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "[Loader:upload_to_device] cudaMemcpy depth_host -> depth_device FAILED!!!");

        CHECK_CUDA(cudaMalloc(&xyz_map_on_device,
                              input_image_pixel_num * 3 * sizeof(float)),
                   "[Loader:upload_to_device] RefinePreProcess malloc managed `xyz_map_on_device` failed!!!");
        auto converter_ = std::make_shared<Converter>();
        converter_->convert_depth_to_xyz_map(static_cast<float *>(depth_on_device),
                                             input_image_height,
                                             input_image_width,
                                             static_cast<float *>(xyz_map_on_device),
                                             intrinsic_matrix(0, 0),
                                             intrinsic_matrix(1, 1),
                                             intrinsic_matrix(0, 2),
                                             intrinsic_matrix(1, 2),
                                             0.1);
        return true;
    }
};
