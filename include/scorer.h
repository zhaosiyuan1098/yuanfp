#pragma once

#include "utils.h"

class Scorer
{
private:
    /* data */
public:
    Scorer(std::string engine_path)
    {
        Checker checker;
        if (!checker.check_file_suffix(engine_path, ".engine"))
        {
            throw std::invalid_argument("Invalid engine path");
        };
        std::cout << "scorer path: " << engine_path << std::endl;
        std::cout << "Scorer created" << std::endl;
    };
    ~Scorer();
    void scorePreProcess()
    {
        std::cout << "ScorePreProcess" << std::endl;
    }
    void scorePostProcess()
    {
        std::cout << "ScorePostProcess" << std::endl;
    }
};

Scorer::~Scorer()
{
}

std::shared_ptr<Scorer> createScorer(std::string engine_path, const std::unordered_map<std::string, std::vector<int64_t>> &input_blobs_shape,
                                     const std::unordered_map<std::string, std::vector<int64_t>> &output_blobs_shape,
                                     const int mem_buf_size)
{
    Checker checker;
    if (!checker.check_file_suffix(engine_path, ".engine"))
    {
        throw std::invalid_argument("Invalid engine path");
    };
    return std::make_shared<Scorer>(engine_path);
}
