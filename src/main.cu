#include <glog/logging.h>
#include <gtest/gtest.h>
#include <string>
#include "foundationpose.h"
#include "loader.h"
#include "pose.h"
#include "utils.h"

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    google::InitGoogleLogging(argv[0]);

    FLAGS_minloglevel = 0; // 设置日志级别为 0，显示所有级别的日志

    int result = RUN_ALL_TESTS();

    google::ShutdownGoogleLogging();
    return result;
}

TEST(main, Foundationpose) {
    testing::internal::CaptureStdout(); // 开始捕获标准输出

    std::cout << "Hello World!" << std::endl;
    Foundationpose foundationpose;
    Loader loader;
    Pose pose;
    std::string first_rgb_path = "path/to/rgb";
    std::string first_depth_path = "path/to/depth";
    std::string first_mask_path = "path/to/mask";
    auto [rgb, depth, mask] = loader.load_rgb_depth_mask(first_rgb_path, first_depth_path, first_mask_path);

    // 检查数据是否加载成功
    ASSERT_FALSE(rgb.empty()) << "Failed to load RGB image!";
    ASSERT_FALSE(depth.empty()) << "Failed to load depth image!";
    ASSERT_FALSE(mask.empty()) << "Failed to load mask image!";

    LOG(INFO) << "RGB: " << rgb.size() << " Depth: " << depth.size() << " Mask: " << mask.size();
}