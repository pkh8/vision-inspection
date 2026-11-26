#include <iostream>
#include <windows.h>

#include <opencv2/core/version.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <filesystem>

#include "ImageLoader.h"
#include "ImagePreProcessor.h"
#include "DefectDetector.h"
#include "InspectionVisualizer.h"
#include "Inspection.h"
#include "BatchInspectionRunner.h"


int main()
{


    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

	std::cout << "Vision Inspection Start" << std::endl;
	std::cout << "OpenCV version:" << CV_VERSION << std::endl;

    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

    ImageLoader loader;
    ImagePreProcessor preProcessor;
    DefectDetector defectDetector;
    InspectionVisualizer visualizer;

    Inspection inspection(loader, preProcessor, defectDetector, visualizer);
    InspectionConfig config;
    BatchInspectionRunner batchRunner(inspection);

    const std::filesystem::path inputDirectory =
        "K:/Project/AIAgent/Vision_Project/data/sample/testImages";

    const std::filesystem::path outputDirectory =
        "K:/Project/AIAgent/Vision_Project/results/batch";

    batchRunner.run(
        inputDirectory,
        outputDirectory,
        config
    );

	return 0;
}

