cmake_minimum_required(VERSION 2.8.12)

########################################
# Install googletest in local directory
########################################

set(GTEST_DIR ${CMAKE_CURRENT_BINARY_DIR}/googletest-download)

# Download and install GoogleTest
configure_file(
	"${PROJECT_SOURCE_DIR}/cmake/googletest.cmake.in"
	"${GTEST_DIR}/CMakeLists.txt"
	IMMEDIATE @ONLY)

execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" . WORKING_DIRECTORY ${GTEST_DIR} )
execute_process(COMMAND ${CMAKE_COMMAND} --build . WORKING_DIRECTORY ${GTEST_DIR})

# Prevent overriding the parent project's compiler/linker settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to the build. This defines the `gtest` and `gtest_main` targets. Don't add these
# targets to the ALL target so that `make install` will ignore googletest installation targets
add_subdirectory(${GTEST_DIR}/googletest-src ${GTEST_DIR}/googletest-build EXCLUDE_FROM_ALL)

# set(test "hello")
# add_definitions(-DDARKNET_WRAPPER_SOURCE_PATH )
# set(DARKNET_WRAPPER_SOURCE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../")
# message("Tests path dir = ${test}")

########################################
# Define Tests
########################################
add_executable(parameters_test src/parameters_test.cpp)
target_link_libraries(parameters_test gtest_main ${PROJECT_NAME})

add_executable(detection_test src/detection_test.cpp)
target_link_libraries(detection_test gtest_main ${PROJECT_NAME} ${OpenCV_LIBRARIES})

# Run all the tests by running the command: $ make gtests
add_custom_target(gtests parameters_test)


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