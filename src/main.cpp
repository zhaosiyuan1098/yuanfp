#include"string"
#include"foundationpose.h"
#include"loader.h"
#include<gtest/gtest.h>

// int main(){

       
// }

TEST(main, Foundationpose) {
    std::cout << "Hello World!" << std::endl;
    Foundationpose foundationpose;
    Loader loader;
    auto [rgb, depth, mask] = loader.load_rgb_depth_mask(first_rgb_path, first_depth_path, first_mask_path);
}