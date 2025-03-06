#pragma once

#include "utils.h"

class Loader
{
private:
    /* data */
public:
    Loader(/* args */);
    ~Loader();

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
};


Loader::Loader(/* args */)
{
}

Loader::~Loader()
{
}
