#include "Inspection.h"

#include<opencv2/imgcodecs.hpp>


Inspection::Inspection(
    ImageLoader& imageLoader,
    ImagePreProcessor& preProcessor,
    DefectDetector& defectDetector,
    InspectionVisualizer& visuaLizer
)
    :imageLoader_(imageLoader),
    preProcessor_(preProcessor),
    defectDetector_(defectDetector),
    visualizer_(visuaLizer)
{
   
}

InspectionResult Inspection::run(
    const std::string& imagePath,
    const std::string& outputPath,
    const std::string& resultOutputPath,
    const InspectionConfig& config
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
            config.thresholdValue,
            config.morphologyKernelSize
        );

    if (binaryImage.empty())
    {
        result.message = "이진화에 실패했습니다.";
        return result;
    }
    
    InspectionResult detectionResult = defectDetector_.defect(binaryImage, config.MinDefectArea);

    if (!detectionResult.success)
    {
        return detectionResult;
    }

    if (!cv::imwrite(outputPath, binaryImage))
    {
        detectionResult.success = false;
        detectionResult.message =
            "이진화 이미지 저장에 실패했습니다.";

        return detectionResult;
    }

    cv::Mat visualizedImage = visualizer_.drawResult(image, detectionResult);

    if (!cv::imwrite(resultOutputPath, visualizedImage))
    {
        detectionResult.success = false;
        detectionResult.message = "결과 이미지 저장에 실패했습니다.";
        return detectionResult;
    }


    return detectionResult;
}
