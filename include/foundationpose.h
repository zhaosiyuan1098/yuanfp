#pragma once

#include "refiner.cuh"
#include "scorer.h"
#include "utils.h"
#include "loader.h"
#include "register.h"
#include "scorer.h"
#include "tracker.h"
#include"converter.cuh"
class Foundationpose
{
private:
    std::shared_ptr<Refiner> refiner_;
    std::shared_ptr<Scorer> scorer_;
    std::shared_ptr<Loader> loader_;
    Eigen::Matrix3f intrinsic_matrix;
    std::shared_ptr<Register> register_;
    std::shared_ptr<Tracker> tracker_;

    int input_image_height = MAX_INPUT_IMAGE_HEIGHT;
    int input_image_width = MAX_INPUT_IMAGE_WIDTH;
    int crop_window_height = 160;
    int crop_window_width = 160;
    float min_depth = 0.1;
    float max_depth = 4.0;

public:
    Foundationpose(/* args */)
    {
        this->refiner_ = std::make_shared<Refiner>(refiner_engine_path_);
        this->scorer_ = std::make_shared<Scorer>(scorer_engine_path_);
        auto loader_ = std::make_shared<Loader>();
        this->intrinsic_matrix = loader_->load_camK();
    };
    ~Foundationpose()=default;

    void regist() { register_->regist(); }

    void track()
    {
        tracker_->track();
    }
};
