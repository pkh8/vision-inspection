#include "InspectionVisualizer.h"
#include "RoiExtractor.h"
#include <opencv2/imgproc.hpp>

cv::Mat InspectionVisualizer::drawResult(const cv::Mat& image, const InspectionResult& result, const InspectionConfig& config) const
{
    cv::Mat resultImage = image.clone();
    const RoiExtractor roiExtractor(config);

    for (const cv::Rect& region : result.defectRegions)
    {
        cv::rectangle(
            resultImage,
            region,
            cv::Scalar(0, 0, 255),
            2
        );

        const cv::Rect expandedRoi = roiExtractor.makeExpandedSquareRoi(
            region,
            resultImage.cols,
            resultImage.rows
        );

        if (!expandedRoi.empty())
        {
            cv::rectangle(
                resultImage,
                expandedRoi,
                cv::Scalar(255, 200, 0),
                1
            );
        }
    }

    return resultImage;
}
