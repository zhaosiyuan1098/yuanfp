#pragma once

#include <iostream>

class Checker
{
private:
    /* data */
public:
    Checker(/* args */)
    {
        std::cout << "Checker created" << std::endl;
    };
    ~Checker() = default;
    bool check_file_suffix(const std::string &file_path, const std::string &suffix)
    {
        const size_t mark = file_path.rfind('.');
        std::string suf;
        return mark != file_path.npos &&
               (suf = file_path.substr(mark, file_path.size() - mark)) == suffix;
    };
};
