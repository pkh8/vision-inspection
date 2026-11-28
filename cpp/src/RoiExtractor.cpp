#include "RoiExtractor.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <opencv2/imgcodecs.hpp>

namespace
{
    std::string formatRoiFileName(const std::string& baseName, int index)
    {
        std::ostringstream oss;
        oss << baseName << "_roi_" << std::setw(2) << std::setfill('0') << index << ".png";
        return oss.str();
    }
} // namespace

RoiExtractor::RoiExtractor(const InspectionConfig& config)
    : config_(config)
{
    config_.minRoiSide = std::max(1, config_.minRoiSide);
}

cv::Rect RoiExtractor::makeExpandedSquareRoi(
    const cv::Rect& box,
    int imageWidth,
    int imageHeight
) const
{
    if (imageWidth <= 0 || imageHeight <= 0 || box.empty())
    {
        return cv::Rect();
    }

    const int padW = static_cast<int>(std::round(box.width * config_.roiPaddingRatio));
    const int padH = static_cast<int>(std::round(box.height * config_.roiPaddingRatio));

    const int paddedW = box.width + padW * 2;
    const int paddedH = box.height + padH * 2;
    const int maxSide = std::min(imageWidth, imageHeight);
    const int side = std::min(std::max(paddedW, paddedH), maxSide);

    const int cx = box.x + box.width / 2;
    const int cy = box.y + box.height / 2;

    int x = cx - side / 2;
    int y = cy - side / 2;

    x = std::max(0, std::min(x, imageWidth - side));
    y = std::max(0, std::min(y, imageHeight - side));

    return cv::Rect(x, y, side, side);
}

std::vector<RoiRecord> RoiExtractor::extractAndSave(
    const cv::Mat& sourceImage,
    const std::filesystem::path& sourceFilePath,
    const std::vector<cv::Rect>& defectRegions,
    const std::filesystem::path& roiOutputDir
) const
{
    std::vector<RoiRecord> savedRecords;

    if (sourceImage.empty() || defectRegions.empty())
    {
        return savedRecords;
    }

    const cv::Rect imageBounds(0, 0, sourceImage.cols, sourceImage.rows);
    const std::string stemName = sourceFilePath.stem().string();
    const std::string sourceFileName = sourceFilePath.filename().string();
    bool dirCreated = false;

    for (size_t i = 0; i < defectRegions.size(); ++i)
    {
        const cv::Rect expanded = makeExpandedSquareRoi(
            defectRegions[i],
            sourceImage.cols,
            sourceImage.rows
        );
        const cv::Rect safeBox = expanded & imageBounds;

        if (safeBox.width < config_.minRoiSide || safeBox.height < config_.minRoiSide)
        {
            continue;
        }

        if (!dirCreated)
        {
            std::filesystem::create_directories(roiOutputDir);
            dirCreated = true;
        }

        const int roiIdx = static_cast<int>(savedRecords.size());
        const std::string roiFileName = formatRoiFileName(stemName, roiIdx);
        const std::filesystem::path savePath = roiOutputDir / roiFileName;

        const cv::Mat cropped = sourceImage(safeBox).clone();
        if (!cv::imwrite(savePath.string(), cropped))
        {
            continue;
        }

        RoiRecord record;
        record.roiFileName = roiFileName;
        record.sourceFileName = sourceFileName;
        record.roiIndex = roiIdx;
        record.rawBox = defectRegions[i];
        record.squareRoi = safeBox;
        savedRecords.push_back(record);
    }

    return savedRecords;
}

void RoiExtractor::appendManifest(
    const std::filesystem::path& manifestPath,
    const std::vector<RoiRecord>& records
) const
{
    if (records.empty())
    {
        return;
    }

    const bool needHeader =
        !std::filesystem::exists(manifestPath) ||
        std::filesystem::file_size(manifestPath) == 0;

    std::ofstream file(
        manifestPath.string(),
        std::ios::out | std::ios::app | std::ios::binary
    );
    if (!file.is_open())
    {
        return;
    }

    if (needHeader)
    {
        const unsigned char utf8Bom[] = { 0xEF, 0xBB, 0xBF };
        file.write(reinterpret_cast<const char*>(utf8Bom), sizeof(utf8Bom));
        file << "roi_file,source_file,roi_index,raw_x,raw_y,raw_w,raw_h,roi_x,roi_y,roi_side\n";
    }

    for (const auto& rec : records)
    {
        file << rec.roiFileName << ","
            << rec.sourceFileName << ","
            << rec.roiIndex << ","
            << rec.rawBox.x << ","
            << rec.rawBox.y << ","
            << rec.rawBox.width << ","
            << rec.rawBox.height << ","
            << rec.squareRoi.x << ","
            << rec.squareRoi.y << ","
            << rec.squareRoi.width << "\n";
    }
}
