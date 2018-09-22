cmake_minimum_required(VERSION 2.8.12)
project(darknet_wrapper)

# Set c++11 cmake flags
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

# Define path of darknet folder here.
find_path(DARKNET_PATH
    NAMES "README.md"
    HINTS "${CMAKE_CURRENT_SOURCE_DIR}/darknet/")
message("Darknet path dir = ${DARKNET_PATH}")
add_definitions(-DDARKNET_FILE_PATH="${DARKNET_PATH}")

# Find CUDA
FIND_PACKAGE(CUDA)
if (CUDA_FOUND)
    find_package(CUDA REQUIRED)
    message(STATUS "CUDA: found!")
    message(STATUS "CUDA Version: ${CUDA_VERSION_STRINGS}")
    message(STATUS "CUDA Libararies: ${CUDA_LIBRARIES}")
    set(CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};
        -O3
        -gencode arch=compute_30,code=sm_30
        -gencode arch=compute_35,code=sm_35
        -gencode arch=compute_50,code=[sm_50,compute_50]
        -gencode arch=compute_52,code=[sm_52,compute_52]
        -gencode arch=compute_61,code=sm_61
        -gencode arch=compute_62,code=sm_62
    )
    add_definitions(-DGPU)
else()
    message(STATUS "CUDA: not found.")
    list(APPEND LIBRARIES "m")
endif()

# Find rquired packeges
find_package(Boost REQUIRED COMPONENTS thread)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)


# Enable OPENCV in darknet
add_definitions(-DOPENCV)
add_definitions(-O4 -g)

include_directories(
    ${BOOST_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIR}
    ${DARKNET_PATH}/src
    ${DARKNET_PATH}/include
    include
)



set (SRCS 
    src/YoloObjectDetector.cpp
    src/image_interface.c
    src/common/dynamic_params.cpp

    ${DARKNET_PATH}/src/activation_layer.c        ${DARKNET_PATH}/src/im2col.c
    ${DARKNET_PATH}/src/activations.c             ${DARKNET_PATH}/src/image.c
    ${DARKNET_PATH}/src/avgpool_layer.c           ${DARKNET_PATH}/src/layer.c
    ${DARKNET_PATH}/src/batchnorm_layer.c         ${DARKNET_PATH}/src/list.c
    ${DARKNET_PATH}/src/blas.c                    ${DARKNET_PATH}/src/local_layer.c
    ${DARKNET_PATH}/src/box.c                     ${DARKNET_PATH}/src/lstm_layer.c
    ${DARKNET_PATH}/src/col2im.c                  ${DARKNET_PATH}/src/matrix.c
    ${DARKNET_PATH}/src/connected_layer.c         ${DARKNET_PATH}/src/maxpool_layer.c
    ${DARKNET_PATH}/src/convolutional_layer.c     ${DARKNET_PATH}/src/network.c
    ${DARKNET_PATH}/src/cost_layer.c              ${DARKNET_PATH}/src/normalization_layer.c
    ${DARKNET_PATH}/src/crnn_layer.c              ${DARKNET_PATH}/src/option_list.c
    ${DARKNET_PATH}/src/crop_layer.c              ${DARKNET_PATH}/src/parser.c
    ${DARKNET_PATH}/src/cuda.c                    ${DARKNET_PATH}/src/region_layer.c
    ${DARKNET_PATH}/src/data.c                    ${DARKNET_PATH}/src/reorg_layer.c
    ${DARKNET_PATH}/src/deconvolutional_layer.c   ${DARKNET_PATH}/src/rnn_layer.c
    ${DARKNET_PATH}/src/demo.c                    ${DARKNET_PATH}/src/route_layer.c
    ${DARKNET_PATH}/src/detection_layer.c         ${DARKNET_PATH}/src/shortcut_layer.c
    ${DARKNET_PATH}/src/dropout_layer.c           ${DARKNET_PATH}/src/softmax_layer.c
    ${DARKNET_PATH}/src/gemm.c                    ${DARKNET_PATH}/src/tree.c
    ${DARKNET_PATH}/src/gru_layer.c               ${DARKNET_PATH}/src/utils.c
    ${DARKNET_PATH}/src/upsample_layer.c          ${DARKNET_PATH}/src/logistic_layer.c
    ${DARKNET_PATH}/src/l2norm_layer.c            ${DARKNET_PATH}/src/yolo_layer.c

    ${DARKNET_PATH}/examples/art.c                ${DARKNET_PATH}/examples/lsd.c
    ${DARKNET_PATH}/examples/attention.c          ${DARKNET_PATH}/examples/nightmare.c
    ${DARKNET_PATH}/examples/captcha.c            ${DARKNET_PATH}/examples/regressor.c
    ${DARKNET_PATH}/examples/cifar.c              ${DARKNET_PATH}/examples/rnn.c
    ${DARKNET_PATH}/examples/classifier.c         ${DARKNET_PATH}/examples/segmenter.c
    ${DARKNET_PATH}/examples/coco.c               ${DARKNET_PATH}/examples/super.c
    ${DARKNET_PATH}/examples/darknet.c            ${DARKNET_PATH}/examples/tag.c
    ${DARKNET_PATH}/examples/detector.c           ${DARKNET_PATH}/examples/yolo.c
    ${DARKNET_PATH}/examples/go.c
)

set (LINK_LIBS 
    m
    pthread
    stdc++
    ${Boost_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${OpenCV_LIBS}
    ${YAML_CPP_LIBRARIES}
)

if (CUDA_FOUND)

    link_directories( /usr/local/cuda/lib64 )

    set (SRCS 
        ${SRCS}
        ${DARKNET_PATH}/src/activation_kernels.cu     ${DARKNET_PATH}/src/crop_layer_kernels.cu
        ${DARKNET_PATH}/src/avgpool_layer_kernels.cu  ${DARKNET_PATH}/src/deconvolutional_kernels.cu
        ${DARKNET_PATH}/src/blas_kernels.cu           ${DARKNET_PATH}/src/dropout_layer_kernels.cu
        ${DARKNET_PATH}/src/col2im_kernels.cu         ${DARKNET_PATH}/src/im2col_kernels.cu
        ${DARKNET_PATH}/src/convolutional_kernels.cu  ${DARKNET_PATH}/src/maxpool_layer_kernels.cu
    )

    cuda_add_library(${PROJECT_NAME} SHARED ${SRCS} )

    set (LINK_LIBS
        ${LINK_LIBS}
        cuda
        cudart
        cublas
        curand)


else()
    add_library(${PROJECT_NAME} SHARED ${SRCS} )


endif()




target_link_libraries(${PROJECT_NAME} ${LINK_LIBS})


# set(INSTALL_LIB_DIR     lib                      CACHE PATH "Installation directory for libraries")
# set(INSTALL_BIN_DIR     bin                      CACHE PATH "Installation directory for executables")
# set(INSTALL_INCLUDE_DIR include                  CACHE PATH "Installation directory for header files")
# set(INSTALL_CMAKE_DIR   lib/cmake/darknet_wrapper   CACHE PATH "Installation directory for CMake Config files")


# target_include_directories(${PROJECT_NAME} PUBLIC 
#               "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
#               "$<INSTALL_INTERFACE:${INSTALL_INCLUDE_DIR}>")

# Let other projects see this project
# add targets to an export group
# install(TARGETS ${PROJECT_NAME}
#         EXPORT ${PROJECT_NAME}-export
#         LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/libs
#         ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
# export(TARGETS ${PROJECT_NAME} FILE ${PROJECT_NAME}-exports.cmake)
# export(PACKAGE ${PROJECT_NAME} )
# message(STATUS "current source dir ${CMAKE_CURRENT_SOURCE_DIR}")

# set(darknet_wrapper_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include")


# message(STATUS "Install lib dir ${INSTALL_LIB_DIR}")

# install(TARGETS ${PROJECT_NAME}
#     EXPORT  ${PROJECT_NAME}
#     RUNTIME DESTINATION "${INSTALL_LIB_DIR}" # neccessary for WIN32
#     LIBRARY DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/lib")

# Actually place the `find_package` instructions on the user's system
# install(EXPORT ${PROJECT_NAME} DESTINATION "${INSTALL_CMAKE_DIR}")

# export(EXPORT ${PROJECT_NAME} FILE ${PROJECT_NAME}-config.cmake)

# install(EXPORT my_library DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/lib")

##################################
#  Unit Testing with GoogleTest  #
##################################


option(BUILD_TESTS "Build the darknet_wrapper GTests" OFF)
# message(STATUS "current source ${CMAKE_CURRENT_SOURCE_DIR}")
# find_path(WRAPPER_FILE_PATH
#     HINTS "${CMAKE_CURRENT_SOURCE_DIR}/../darknet_wrapper")
add_definitions( -DWRAPPER_FILE_PATH="${CMAKE_CURRENT_SOURCE_DIR}")
# set(WRAPPER_FILE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

if(BUILD_TESTS)
message(STATUS "HERE")

  add_subdirectory(tests)
endif()


