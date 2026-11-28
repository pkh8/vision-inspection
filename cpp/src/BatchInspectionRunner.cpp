#include "BatchInspectionRunner.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <opencv2/imgcodecs.hpp>

BatchInspectionRunner::BatchInspectionRunner(
    Inspection& inspection,
    const RoiExtractor* roiExtractor
)
    : inspection_(inspection)
    , roiExtractor_(roiExtractor)
{
}

bool BatchInspectionRunner::isImageFile(
    const std::filesystem::path& filePath
) const
{
    if (!filePath.has_extension())
    {
        return false;
    }

    std::string extension = filePath.extension().string();
    std::transform(
        extension.begin(),
        extension.end(),
        extension.begin(),
        [](unsigned char character)
        {
            return static_cast<char>(std::tolower(character));
        }
    );

    return extension == ".jpg"
        || extension == ".jpeg"
        || extension == ".png"
        || extension == ".bmp";
}

std::filesystem::path BatchInspectionRunner::makeOutputPath(
    const std::filesystem::path& outputDirectory,
    const std::filesystem::path& inputFile,
    const std::string& suffix
) const
{
    const std::string baseName = inputFile.stem().string();
    return outputDirectory / (baseName + suffix + ".jpg");
}

void BatchInspectionRunner::run(
    const std::filesystem::path& inputDirectory,
    const std::filesystem::path& outputDirectory,
    const InspectionConfig& config
)
{
    if (!std::filesystem::exists(inputDirectory)
        || !std::filesystem::is_directory(inputDirectory))
    {
        std::cout << "입력 폴더를 찾을 수 없습니다." << inputDirectory << std::endl;
        return;
    }

    std::filesystem::create_directories(outputDirectory);

    const std::filesystem::path summaryPath = outputDirectory / "summary.csv";
    std::ofstream csvFile(
        summaryPath.string(),
        std::ios::out | std::ios::binary
    );

    if (!csvFile.is_open())
    {
        std::cout << "CSV파일을 열 수 없습니다." << std::endl;
        return;
    }

    const unsigned char utf8Bom[] = { 0xEF, 0xBB, 0xBF };
    csvFile.write(reinterpret_cast<const char*>(utf8Bom), sizeof(utf8Bom));
    csvFile << "file_name,success,passed,defect_count,message\n";

    const std::filesystem::path roiDir = outputDirectory / "train_rois";
    const std::filesystem::path manifestPath = outputDirectory / "roi_manifest.csv";

    for (const auto& entry : std::filesystem::directory_iterator(inputDirectory))
    {
        if (!entry.is_regular_file())
        {
            continue;
        }

        const auto& inputFile = entry.path();
        if (!isImageFile(inputFile))
        {
            continue;
        }

        const auto binaryPath = makeOutputPath(outputDirectory, inputFile, "_binary");
        const auto resultPath = makeOutputPath(outputDirectory, inputFile, "_result");

        InspectionResult result = inspection_.run(
            inputFile.string(),
            binaryPath.string(),
            resultPath.string(),
            config
        );

        if (roiExtractor_ != nullptr && result.success && !result.defectRegions.empty())
        {
            const cv::Mat sourceImage = cv::imread(inputFile.string(), cv::IMREAD_UNCHANGED);
            if (!sourceImage.empty())
            {
                const auto records = roiExtractor_->extractAndSave(
                    sourceImage,
                    inputFile,
                    result.defectRegions,
                    roiDir
                );
                roiExtractor_->appendManifest(manifestPath, records);
            }
        }

        csvFile << inputFile.filename().string() << ","
            << result.success << ","
            << result.passed << ","
            << result.defectCount << ","
            << result.message
            << "\n";

        std::cout << inputFile.filename().string()
            << ":" << result.message
            << "/결함 개수:" << result.defectCount
            << std::endl;
    }

    csvFile.close();

    std::cout
        << "CSV 결과 저장 완료: "
        << summaryPath.string()
        << std::endl;
}
