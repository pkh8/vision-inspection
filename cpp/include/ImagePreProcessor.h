#pragma once

#include <opencv2/core.hpp>



class ImagePreProcessor
{
public:

    cv::Mat toGray(const cv::Mat& image) const;

    cv::Mat threshold(
        const cv::Mat& grayImage,
        double thresholdValue = 100.0
    )const;

};
