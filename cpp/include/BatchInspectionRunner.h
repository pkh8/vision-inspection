#pragma once

#include <filesystem>
#include <string>

#include "Inspection.h"
#include "InspectionConfig.h"

class BatchInspectionRunner
{
public:
    explicit BatchInspectionRunner(
        Inspection& inspection
    );

    // 입력 폴더의 이미지들을 순서대로 검사한다.
    void run(
        const std::filesystem::path& inputDirectory,
        const std::filesystem::path& outputDirectory,
        const InspectionConfig& config
    );

private:
    // 지원하는 이미지 확장자인지 확인한다.
    bool isImageFile(
        const std::filesystem::path& filePath
    ) const;

    // 검사 결과 파일의 저장 경로를 만든다.
    std::filesystem::path makeOutputPath(
        const std::filesystem::path& outputDirectory,
        const std::filesystem::path& inputFile,
        const std::string& suffix
    ) const;

    // 외부에서 전달받은 검사 객체를 보관한다.
    Inspection& inspection_;
};
