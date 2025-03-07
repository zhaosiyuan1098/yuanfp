#pragma once

#include "utils.h"
#include "dataKeeper.h"

class Preprocessor
{
private:
    /* data */
public:
    Preprocessor(/* args */) = default;
    ~Preprocessor() = default;
    RefinePreProcess(std::shared_ptr<DataKeeper> datakeeper)
    {
        auto datakeeper = std::dynamic_pointer_cast<DataKeeper>(datakeeper);
        CHECK_STATE(datakeeper != nullptr,
                    "[Preprocessor:RefinePreProcess] RefinePreProcess Got INVALID package ptr");
    }
};
