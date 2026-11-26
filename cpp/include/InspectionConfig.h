#pragma once

struct InspectionConfig
{
    //밝기 이진화 기준값
    double thresholdValue = 75.0;

    //결함으로 인정할 최소 컨투어 면적
    double MinDefectArea = 20.0;

    int morphologyKernelSize = 5;

};
