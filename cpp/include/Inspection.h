#pragma once

#include "ImageLoader.h"
#include "ImagePreProcessor.h"
#include "InspectionResult.h"
#include "DefectDetector.h"
#include "InspectionVisualizer.h"
#include "InspectionConfig.h"


#include <string>

class Inspection
{
public:
    Inspection(
        ImageLoader& imageLoader,
        ImagePreProcessor& preProcessor,
        DefectDetector& defectDetector,
        InspectionVisualizer& visuaLizer
    );


    InspectionResult run(
        const std::string& imagePath,
        const std::string& outputPath,
        const std::string& resultOutputPath,
        const InspectionConfig& config
    );


private:
    ImageLoader& imageLoader_;
    ImagePreProcessor& preProcessor_;
    DefectDetector& defectDetector_;
    InspectionVisualizer& visualizer_;
};
