#include <iostream>
#include <windows.h>

#include <opencv2/core/version.hpp>
#include <opencv2/core/utils/logger.hpp>

#include "ImageLoader.h"
#include "ImagePreProcessor.h"
#include "DefectDetector.h"
#include "InspectionVisualizer.h"
#include "Inspection.h"
#include "InspectionResult.h"


int main()
{

    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
	std::cout << "Vision Inspection Start" << std::endl;
	std::cout << "OpenCV version:" << CV_VERSION << std::endl;
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

	const std::string imagePath = "K:/Project/AIAgent/Vision_Project/data/sample/data/images/5fe0a834387b43f0972178e926fe63ff.jpg";

    const std::string outputPath =
        "K:/Project/AIAgent/Vision_Project/results/binary.jpg";

    const std::string resultOutputPath =
        "K:/Project/AIAgent/Vision_Project/results/inspection_result.jpg";


    ImageLoader loader;
    ImagePreProcessor preProcessor;
    DefectDetector defectDetector;
    InspectionVisualizer visualizer;

    Inspection inspection(loader, preProcessor, defectDetector, visualizer);
    double threshold = 75;

    InspectionResult result = inspection.run(imagePath, outputPath, resultOutputPath, threshold);

    if (!result.success)
    {
        std::cout << "검사 실패:" << result.message << std::endl;
        return 1;
    }
    

    std::cout << "검사 실행 성공" << std::endl;
    std::cout << result.message << std::endl;
    std::cout << "결함 개수: "
        << result.defectCount
        << std::endl;


	return 0;
}

