cmake_minimum_required(VERSION 3.20)

project(VisionInspection)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/utf-8)
endif()

find_package(OpenCV REQUIRED)

add_executable(
    VisionInspection
    src/main.cpp
    src/ImageLoader.cpp
    src/ImagePreProcessor.cpp
    src/Inspection.cpp
    src/DefectDetector.cpp
    src/InspectionVisualizer.cpp
    src/BatchInspectionRunner.cpp
    src/RoiExtractor.cpp

    include/ImageLoader.h
    include/ImagePreProcessor.h
    include/Inspection.h
    include/InspectionConfig.h
    include/InspectionResult.h
    include/DefectDetector.h
    include/InspectionVisualizer.h
    include/BatchInspectionRunner.h
    include/RoiExtractor.h
    
)


target_include_directories(
    VisionInspection
    PRIVATE include
)

target_link_libraries(
    VisionInspection
    PRIVATE ${OpenCV_LIBS}
)
