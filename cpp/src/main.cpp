#include <iostream>
#include <windows.h>
#include <opencv2/opencv.hpp>
#include "ImageLoader.h"
#include "ImagePreprocessor.h"


int main()
{

    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
	std::cout << "Vision Inspection Start" << std::endl;
	std::cout << "OpenCV version:" << CV_VERSION << std::endl;

	const std::string imagePath = "K:/Project/AIAgent/Vision_Project/data/sample/data/images/00a7233a4d564357844d59b7b38aa85a.jpg";

    const std::string outputPath =
        "K:/Project/AIAgent/Vision_Project/results/binary.jpg";


    // 이미지 입력 담당 객체
    ImageLoader loader;

    cv::Mat image = loader.load(imagePath);


	// 이미지 읽기 실패 여부 확인
	if (image.empty())
	{
		std::cout << "이미지를 읽지 못했습니다." << std::endl;
		std::cout << "imageread:" << imagePath << std::endl;
        return 1;
	}
	std::cout << "이미지 읽기 성공" << std::endl;
	std::cout << "가로: " << image.cols << std::endl;
	std::cout << "세로: " << image.rows << std::endl;
	std::cout << "채널 수: " << image.channels() << std::endl;


    // 이미지 전처리 담당 객체
    ImagePreProcessor preProcessor;
    cv::Mat grayImage = preProcessor.toGray(image);


    if (grayImage.empty())
    {
        std::cout << "그레이스케일 변환에 실패했습니다." << std::endl;
        return 1;
    }

    std::cout << "그레이스케일 변환 성공" << std::endl;
    std::cout   << "변환 후 채널 수 : "
                << grayImage.channels()
                << std::endl;


    cv::Mat binaryImage = preProcessor.threshold(grayImage, 100.0);

    if (binaryImage.empty())
    {
        std::cout << "이진화에 실패했습니다." << std::endl;
        return 1;
    }

    const bool saved =
        cv::imwrite(outputPath, binaryImage);

    if (!saved)
    {
        std::cout << "이진화 이미지 저장에 실패했습니다." << std::endl;
        return 1;
    }

    std::cout << "이진화 이미지 저장 성공" << std::endl;
    std::cout << "저장 경로: " << outputPath << std::endl;

	return 0;
}

