#include "ImagePreProcessor.h"
#include <opencv2/imgproc.hpp>


cv::Mat ImagePreProcessor::toGray(const cv::Mat& image) const
{
    //입력 이미지가 비어 있으면 변환하지 않고 빈 결과 반환
    if (image.empty())
    {
        return cv::Mat();
    }


    cv::Mat grayImage;

    // 컬러 RGB이미지를 1채널 그레이스케일로 변환
    cv::cvtColor(
        image,
        grayImage,
        cv::COLOR_BGR2GRAY
    );
    
    return grayImage;
}


cv::Mat ImagePreProcessor::threshold(
    const cv::Mat& grayImage,
    double thresholdValue,
    int kernelSize
)const
{
    //입력 이미지가 비어 있으면 이진화지 않습니다.
    if (grayImage.empty())
    {
        return cv::Mat();
    }

    cv::Mat binaryImage;

    // 기준값보다 밝은 픽셀은 255, 나머지는 0으로 변환합니다.
    cv::threshold(
        grayImage,
        binaryImage,
        thresholdValue,              // 기준 밝기
        255.0,              // 검출 영역을 흰색으로
        cv::THRESH_BINARY_INV
    );

    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, 
        cv::Size(kernelSize, kernelSize));

    cv::morphologyEx(
        binaryImage,
        binaryImage,
        cv::MORPH_CLOSE,
        kernel
    );





    return binaryImage;
}
