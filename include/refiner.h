#pragma once

#include "utils.h"
#include "checker.h"

class Refiner
{
private:
    /* data */
public:
    Refiner(std::string engine_path)
    {
        Checker checker;
        if (!checker.check_file_suffix(engine_path, ".engine"))
        {
            throw std::invalid_argument("Invalid engine path");
        };
        std::cout << "Refiner created" << std::endl;
    };
    ~Refiner() {

    };
};

std::shared_ptr<Refiner> createRefiner(std::string engine_path, const std::unordered_map<std::string, std::vector<int64_t>> &input_blobs_shape,
                                       const std::unordered_map<std::string, std::vector<int64_t>> &output_blobs_shape,
                                       const int mem_buf_size)
{
    Checker checker;
    if (!checker.check_file_suffix(engine_path, ".engine"))
    {
        throw std::invalid_argument("Invalid engine path");
    };
    return std::make_shared<Refiner>(engine_path);
}