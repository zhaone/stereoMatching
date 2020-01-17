cmake_minimum_required (VERSION 2.8)
project (stereoMatching)
# export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/usr/local/include/opencv4
aux_source_directory(. DIR_SRCS)

find_package( OpenCV REQUIRED )
add_executable(main ${DIR_SRCS})
target_link_libraries( main ${OpenCV_LIBS} )