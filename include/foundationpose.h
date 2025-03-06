#pragma once

#include "refiner.h"
#include "scorer.h"
#include "utils.h"

class foundationpose
{
private:
    /* data */
public:
    foundationpose(/* args */);
    ~foundationpose();
    bool UploadDataToDevice(const cv::Mat &rgb,
                            const cv::Mat &depth,
                            const cv::Mat &mask,
                            const std::shared_ptr<int> &package) {}
    bool RefinePreProcess(std::shared_ptr<int> package);

    bool ScorePreprocess(std::shared_ptr<int> package);

    bool ScorePostProcess(std::shared_ptr<int> package);

    bool TrackPostProcess(std::shared_ptr<int> package);
};

foundationpose::foundationpose(/* args */)
{
    auto refiner = std::make_shared<Refiner>();
    auto scorer = std::make_shared<Scorer>();
    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
}

foundationpose::~foundationpose()
{
}
