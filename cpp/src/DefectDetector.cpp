#include "DefectDetector.h"
#include<opencv2/imgproc.hpp>
/*
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
*/


namespace
{
    // 잡티/꼬리를 끊어내기 위한 최소 덩어리 비율 (가장 큰 컨투어 대비 15% 미만은 노이즈로 취급)
    constexpr double kMinAreaRatioToMax = 0.15;
    constexpr int kMinSideLength = 4;
} // namespace

InspectionResult DefectDetector::defect(const cv::Mat& binaryImage, double minArea) const
{
    InspectionResult result;

    if (binaryImage.empty())
    {
        result.message = "검사할 이미지가 비어 있습니다.";
        return result;
    }

    // 1. 얇은 돌기/꼬리/점 노이즈 제거 (커널 크기는 결함 스케일에 맞게 조절)
    cv::Mat cleaned;
    const cv::Mat openKernel =
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(binaryImage, cleaned, cv::MORPH_OPEN, openKernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(cleaned, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 2. 1차로 절대 면적 + 경계 접촉 필터를 통과한 후보와 최대 면적 계산
    struct Candidate
    {
        double area;
        cv::Rect rect;
    };

    std::vector<Candidate> candidates;
    candidates.reserve(contours.size());
    double maxArea = 0.0;

    for (const auto& contour : contours)
    {
        const double area = cv::contourArea(contour);
        if (area < minArea)
        {
            continue;
        }

        const cv::Rect region = cv::boundingRect(contour);
        if (region.width < kMinSideLength || region.height < kMinSideLength)
        {
            continue;
        }

        const bool touchesBorder =
            region.x == 0 ||
            region.y == 0 ||
            region.x + region.width >= cleaned.cols ||
            region.y + region.height >= cleaned.rows;

        if (touchesBorder)
        {
            continue;
        }

        candidates.push_back(Candidate{ area, region });
        if (area > maxArea)
        {
            maxArea = area;
        }
    }

    // 3. 가장 큰 결함 덩어리 대비 너무 작은 파편/노이즈 박스 제거
    const double relativeMinArea = maxArea * kMinAreaRatioToMax;

    for (const auto& cand : candidates)
    {
        if (cand.area < relativeMinArea)
        {
            continue;
        }
        result.defectRegions.push_back(cand.rect);
    }

    result.success = true;
    result.defectCount = static_cast<int>(result.defectRegions.size());
    result.passed = (result.defectCount == 0);
    result.message = result.passed
        ? "결함이 검출되지 않았습니다."
        : "결함 후보가 검출되었습니다.";

    return result;
}
