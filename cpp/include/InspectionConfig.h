#pragma once

struct InspectionConfig
{
    //밝기 이진화 기준값
    double thresholdValue = 82.0;

    //결함으로 인정할 최소 컨투어 면적
    double MinDefectArea = 20.0;

    // 모폴로지 커널 크기
    int morphologyKernelSize = 5;

    // ROI 크롭 여백 비율
    double roiPaddingRatio = 0.30;

    // 최소 정사각 ROI 한 변 크기(px)
    int minRoiSide = 32;

};
