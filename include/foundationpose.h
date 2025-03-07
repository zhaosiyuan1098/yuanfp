#pragma once

#include "refiner.h"
#include "scorer.h"
#include "utils.h"
#include "loader.h"

class Foundationpose
{
private:
    std::shared_ptr<Refiner> refiner_;
    std::shared_ptr<Scorer> scorer_;
    std::shared_ptr<Loader> loader_;
    Eigen::Matrix3f intrinsic_matrix;


    int input_image_height = MAX_INPUT_IMAGE_HEIGHT;
    int input_image_width = MAX_INPUT_IMAGE_WIDTH;
    int crop_window_height = 160;
    int crop_window_width = 160;
    float min_depth = 0.1;
    float max_depth = 4.0;
public:
    Foundationpose(/* args */);
    ~Foundationpose();
    bool UploadDataToDevice(const cv::Mat &rgb,
                            const cv::Mat &depth,
                            const cv::Mat &mask,
                            const std::shared_ptr<int> &package) {}
    bool RefinePreProcess(std::shared_ptr<int> package);

    bool ScorePreprocess(std::shared_ptr<int> package);

    bool ScorePostProcess(std::shared_ptr<int> package);

    bool TrackPostProcess(std::shared_ptr<int> package);

    bool infer(){
        
    };
};

Foundationpose::Foundationpose(/* args */)
{
    this->refiner_ = std::make_shared<Refiner>(refiner_engine_path_);
    this->scorer_ = std::make_shared<Scorer>(scorer_engine_path_);
    auto loader_ = std::make_shared<Loader>();
    this->intrinsic_matrix=loader_->load_camK();

}

Foundationpose::~Foundationpose()
{
}
