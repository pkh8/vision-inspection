#include "InspectionVisualizer.h"
#include <opencv2/imgproc.hpp>



cv::Mat InspectionVisualizer::drawResult(const cv::Mat& image, const InspectionResult& result)const
{

    cv::Mat resultImage = image.clone();

    for (const cv::Rect& region : result.defectRegions)
    {
        cv::rectangle(
            resultImage,
            region,
            cv::Scalar(0, 0, 255),
            2
        );
    }

    return resultImage;
}
