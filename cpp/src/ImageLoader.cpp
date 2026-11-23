#include "ImageLoader.h"


cv::Mat ImageLoader::load(const std::string& imagePath) const
{
    return cv::imread(imagePath, cv::IMREAD_COLOR);
}
