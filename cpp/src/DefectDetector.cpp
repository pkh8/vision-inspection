#include "DefectDetector.h"
#include<opencv2/imgproc.hpp>

InspectionResult DefectDetector::defect(const cv::Mat& binaryImage, double minArea) const
{
    InspectionResult result;

    if (binaryImage.empty())
    {
        result.message =
            "검사할 이미지가 비어 있습니다.";

        return result;
    }

    std::vector<std::vector<cv::Point>> contours;

    cv::findContours(
        binaryImage,
        contours,
        cv::RETR_EXTERNAL,
        cv::CHAIN_APPROX_SIMPLE
    );

    for (const auto& contour : contours)
    {
        const double area =
            cv::contourArea(contour);

        if (area < minArea)
        {
            continue;
        }

        const cv::Rect region =
            cv::boundingRect(contour);

        const bool touchesBorder =
            region.x == 0 ||
            region.y == 0 ||
            region.x + region.width >= binaryImage.cols ||
            region.y + region.height >= binaryImage.rows;

        if (touchesBorder)
        {
            continue;
        }

        result.defectRegions.push_back(region);
    }

    result.success = true;

    result.defectCount =
        static_cast<int>(
            result.defectRegions.size()
            );

    result.passed =
        result.defectCount == 0;

    result.message =
        result.passed
        ? "결함이 검출되지 않았습니다."
        : "결함 후보가 검출되었습니다.";

    return result;
}
