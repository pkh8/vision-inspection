#pragma once

#include <filesystem>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include "InspectionConfig.h"

struct RoiRecord
{
    std::string roiFileName;
    std::string sourceFileName;
    int roiIndex = 0;
    cv::Rect rawBox;
    cv::Rect squareRoi;
};

class RoiExtractor
{
public:
    explicit RoiExtractor(
        const InspectionConfig& config
    );

    cv::Rect makeExpandedSquareRoi(
        const cv::Rect& box,
        int imageWidth,
        int imageHeight
    ) const;

    std::vector<RoiRecord> extractAndSave(
        const cv::Mat& sourceImage,
        const std::filesystem::path& sourceFilePath,
        const std::vector<cv::Rect>& defectRegions,
        const std::filesystem::path& roiOutputDir
    ) const;

    void appendManifest(
        const std::filesystem::path& manifestPath,
        const std::vector<RoiRecord>& records
    ) const;

    double paddingRatio() const noexcept;
    int minRoiSide() const noexcept;

private:
    InspectionConfig config_;


};
