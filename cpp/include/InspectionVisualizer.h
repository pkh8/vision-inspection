#pragma once

#include "InspectionResult.h"
#include"InspectionConfig.h"
#include <opencv2/core.hpp>


class InspectionVisualizer
{
public:
    cv::Mat drawResult(const cv::Mat& image, const InspectionResult& result, const InspectionConfig& config)const;
};
