#include <iostream>
#include <windows.h>
#include <opencv2/core/version.hpp>

#include "ImageLoader.h"
#include "ImagePreProcessor.h"
#include "Inspection.h"
#include "InspectionResult.h"


int main()
{

    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
	std::cout << "Vision Inspection Start" << std::endl;
	std::cout << "OpenCV version:" << CV_VERSION << std::endl;

	const std::string imagePath = "K:/Project/AIAgent/Vision_Project/data/sample/data/images/00a7233a4d564357844d59b7b38aa85a.jpg";

    const std::string outputPath =
        "K:/Project/AIAgent/Vision_Project/results/binary.jpg";


    ImageLoader loader;
    ImagePreProcessor preProcessor;

    Inspection inspection(loader, preProcessor);

    InspectionResult result = inspection.run(imagePath, outputPath, 100.0);

    if (!result.success)
    {
        std::cout << "검사 실패:" << result.message << std::endl;
        return 1;
    }
    

    std::cout << "검사 실행 성공" << std::endl;
    std::cout << result.message << std::endl;


	return 0;
}

