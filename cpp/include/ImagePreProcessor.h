#pragma once

#include <opencv2/core.hpp>
#include<InspectionConfig.h>



class ImagePreProcessor
{
public:

    cv::Mat toGray(const cv::Mat& image) const;

    cv::Mat threshold(
        const cv::Mat& grayImage,
        double thresholdValue,
        int kernelSize
    )const;

};
