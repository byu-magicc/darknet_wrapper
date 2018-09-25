cmake_minimum_required(VERSION 2.8.12)


########################################
# Define Tests
########################################
add_executable(benchmark src/benchmark.cpp)
target_link_libraries(benchmark ${PROJECT_NAME})

message( STATUS "here")

########################################
# Download yolov3 weights
########################################
set(PATH "${CMAKE_CURRENT_SOURCE_DIR}/../yolo_network_config/weights")
set(FILE "${PATH}/yolov3-tiny.weights")
message(STATUS "Checking and downloading yolov3-tiny.weights if needed ...")
if (NOT EXISTS "${FILE}")
    message(STATUS "... file does not exist. Downloading now ...")
    execute_process(COMMAND wget http://pjreddie.com/media/files/yolov3-tiny.weights -P ${PATH})
endif()