#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>


struct InspectionResult
{
    bool success = false;
    bool passed = false;
    int defectCount = 0;
    std::vector<cv::Rect> defectRegions;
    std::string message;
};
