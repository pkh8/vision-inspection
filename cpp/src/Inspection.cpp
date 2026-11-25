#include "Inspection.h"

#include<opencv2/imgcodecs.hpp>


Inspection::Inspection(
    ImageLoader& imageLoder,
    ImagePreProcessor& preProcessor,
    DefectDetector& defectDefector,
    InspectionVisualizer& visualLizer
)
    :imageLoader_(imageLoder),
    preProcessor_(preProcessor),
    defectDetector_(defectDefector),
    visualLizer_(visualLizer)
{
   
}

InspectionResult Inspection::run(
    const std::string& imagePath,
    const std::string& outputPath,
    const std::string& resultOutputPath,
    double thresholdValue
)
{
    InspectionResult result;

    cv::Mat image = imageLoader_.load(imagePath);

    if (image.empty())
    {
        result.message = "이미지를 읽지 못했습니다.";
        return result;
    }

    cv::Mat grayImage =
        preProcessor_.toGray(image);

    if (grayImage.empty())
    {
        result.message = "그레이스케일 변환에 실패했습니다.";
        return result;
    }

    cv::Mat binaryImage =
        preProcessor_.threshold(
            grayImage,
            thresholdValue
        );

    if (binaryImage.empty())
    {
        result.message = "이진화에 실패했습니다.";
        return result;
    }

    InspectionResult detectionResult = defectDetector_.defect(binaryImage, 20.0);

    if (!cv::imwrite(outputPath, binaryImage))
    {
        detectionResult.success = false;
        detectionResult.message =
            "이진화 이미지 저장에 실패했습니다.";

        return detectionResult;
    }


    cv::Mat visualizedImage = visualLizer_.drawResult(image, detectionResult);


    if (!cv::imwrite(resultOutputPath, visualizedImage))
    {
        detectionResult.success = false;
        detectionResult.message = "결과 이미지 저장에 실패했습니다.";
        return detectionResult;
    }

    detectionResult.success = true;

    return detectionResult;
}
