#pragma once

#include "InspectionResult.h"
#include <opencv2/core.hpp>



class DefectDetector
{
public:
    InspectionResult defect(
        const cv::Mat& binaryImage,
        double minArea
    )const;
};
