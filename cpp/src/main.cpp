#include <iostream>
#include <windows.h>
#include <opencv2/opencv.hpp>


int main()
{

    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
	std::cout << "Vision Inspection Start" << std::endl;
	std::cout << "OpenCV version:" << CV_VERSION << std::endl;

	std::string imagePath = "K:/Project/AIAgent/Vision_Project/data/sample/data/images/00a7233a4d564357844d59b7b38aa85a.jpg";

	cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);


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

    cv::Mat grayImage;

    cv::cvtColor(
        image,
        grayImage,
        cv::COLOR_BGR2GRAY
    );

    std::cout << "그레이스케일 변환 성공" << std::endl;
    std::cout   << "변환 후 채널 수 : "
                << grayImage.channels()
                << std::endl;


	return 0;
}

