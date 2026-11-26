#include "BatchInspectionRunner.h"

#include<algorithm>
#include<cctype>
#include<iostream>



BatchInspectionRunner::BatchInspectionRunner(
    Inspection& inspection
)
    : inspection_(inspection)
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
            return static_cast<char>(
                std::tolower(character)
                );
        }
    );

    return extension == ".jpg"
        || extension == ".jpeg"
        || extension == ".png"
        || extension == ".bmp";
}


std::filesystem::path
BatchInspectionRunner::makeOutputPath(
    const std::filesystem::path& outputDirectory,
    const std::filesystem::path& inputFile,
    const std::string& suffix
) const
{
    const std::string baseName =
        inputFile.stem().string();

    return outputDirectory /
        (baseName + suffix + ".jpg");
}



void BatchInspectionRunner::run(
    const std::filesystem::path& inputDirectory,
    const std::filesystem::path& outputDirectory,
    const InspectionConfig& config
)
{
    if(!std::filesystem::exists(inputDirectory)
        || !std::filesystem::is_directory(inputDirectory))
    {
        std::cout << "입력 폴더를 찾을 수 없습니다." << inputDirectory << std::endl;
        return;
    }

    std::filesystem::create_directories(outputDirectory);

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

        std::cout << inputFile.filename().string() << ":" << result.message << "/결함 개수:" << result.defectCount << std::endl;

    }

}



