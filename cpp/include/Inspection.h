#pragma once

#include "ImageLoader.h"
#include "ImagePreProcessor.h"
#include "InspectionResult.h"


#include <string>

class Inspection
{
public:
    Inspection(
        ImageLoader& imageLoader,
        ImagePreProcessor& preProcessor
    );


    InspectionResult run(
        const std::string& imagePath,
        const std::string& outputPath,
        double thresholdValue = 100.0
    );


private:
    ImageLoader& imageLoader_;
    ImagePreProcessor& preProcessor_;
};
