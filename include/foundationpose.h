#pragma once

#include "refiner.h"
#include "scorer.h"
#include "util.h"


class foundationpose
{
private:
    /* data */
public:
    foundationpose(/* args */);
    ~foundationpose();
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
