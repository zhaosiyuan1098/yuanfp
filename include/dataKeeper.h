#pragma once

#include "utils.h"

class DataKeeper
{
private:
    /* data */
public:
    DataKeeper(/* args */);
    ~DataKeeper() = default;
    cv::Mat rgb_on_host;
    // 输入host端depth
    cv::Mat depth_on_host;
    // 输入host端mask
    cv::Mat mask_on_host;
    // 目标物名称
    std::string target_name;

    int input_image_height;

    int input_image_width;

    // device端的输入图像缓存
    std::shared_ptr<void> rgb_on_device;
    // device端的输入深度缓存
    std::shared_ptr<void> depth_on_device;
    // device端由depth转换得到的xyz_map
    std::shared_ptr<void> xyz_map_on_device;
    // device端的输入mask缓存
    // std::shared_ptr<void> mask_on_device;
    // 生成的假设位姿
    std::vector<Eigen::Matrix4f> hyp_poses;
    // refine后的位姿
    std::vector<Eigen::Matrix4f> refine_poses;

    // **最终输出的位姿** //
    Eigen::Matrix4f actual_pose;
};
