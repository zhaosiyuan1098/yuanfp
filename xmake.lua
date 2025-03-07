set_xmakever("2.5.1")

target("simple_tests")
    set_kind("binary")  -- 生成可执行文件
    set_languages("cxx17")  -- 使用 C++17 标准
    add_cxxflags("-O3", "-Wextra", "-Wdeprecated", "-fPIC")  -- 编译选项


    add_files( "src/*.cu")   -- 添加源文件
    add_includedirs("include")  -- 添加头文件搜索路径
    add_includedirs("/usr/include/eigen3")  -- Eigen3 头文件路径
    add_includedirs("/usr/include/opencv4")  -- OpenCV 头文件路径
    add_includedirs("/usr/local/cuda/include")  -- CUDA 头文件路径

    add_packages("gtest", "glog", "opencv", "eigen3", "assimp", "cvcuda", "cuda")

    add_syslinks("pthread", "cuda", "cudart", "nvinfer", "nvonnxparser", "nvcv_types", "cvcuda", "assimp", "glog", "opencv_core", "opencv_imgproc", "opencv_highgui", "opencv_imgcodecs", "opencv_videoio", "gtest", "gtest_main")

    -- 设置库路径
    add_linkdirs("build/linux/x86_64/release")  -- 根据实际路径调整
    add_linkdirs("/usr/local/cuda/lib64")  -- CUDA 库路径
