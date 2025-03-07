#pragma once

#include "utils.h"

class Pose
{
private:
    /* data */
public:
    const Eigen::Vector3f object_dimention;
    Eigen::Matrix4f transformation;
    Pose(/* args */){
        this->transformation<<1,0,0,0,
                        0,1,0,0,
                        0,0,1,0,
                        0,0,0,1;
    };
    ~Pose() =default;
};
