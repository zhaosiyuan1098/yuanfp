#include"utils.h"

class Drawer
{
private:
    /* data */
public:
    Drawer(/* args */);
    ~Drawer();
    void draw3DBoundingBox(cv::Mat &image, Eigen::Matrix3f &camK, Eigen::Matrix4f &pose, Eigen::MatrixXf &vertices, Eigen::MatrixXi &edges){
        std::cout<<"Drawing 3D Bounding Box"<<std::endl;
    };
};

Drawer::Drawer(/* args */)
{
}

Drawer::~Drawer()
{
}
