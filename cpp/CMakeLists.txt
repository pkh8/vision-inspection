cmake_minimum_required(VERSION 3.20)

project(VisionInspection)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
	add_compile_options(/utf-8)
endif()


add_executable(
	VisionInspection
	src/main.cpp
)


find_package(OpenCV REQUIRED)

target_link_libraries(
	VisionInspection
	PRIVATE ${OpenCV_LIBS}
)

