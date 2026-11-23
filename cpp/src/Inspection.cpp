#include "Inspection.h"

#include<opencv2/imgcodecs.hpp>


Inspection::Inspection(
    ImageLoader& imageLoder,
    ImagePreProcessor& preProcessor
)
    :imageLoader_(imageLoder),
    preProcessor_(preProcessor)
{
   
}

InspectionResult Inspection::run(
    const std::string& imagePath,
    const std::string& outputPath,
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

    if (!cv::imwrite(outputPath, binaryImage))
    {
        result.message = "결과 이미지 저장에 실패했습니다.";
        return result;
    }

    result.success = true;
    result.passed = true;
    result.message = "기본 전처리가 완료되었습니다.";

    return result;
}
