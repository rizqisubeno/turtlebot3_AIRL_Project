cmake_minimum_required(VERSION 3.8)
project(tb3_robot)

# Setup the project.
# Its name is defined to be the controller directory name.
get_filename_component(PROJECT ${CMAKE_SOURCE_DIR} NAME)
project(${PROJECT})

#############ROS2_DEPENDENCIES#############
# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# Find Protobuf
#find_package(Protobuf REQUIRED)
#find_package( absl REQUIRED )
include(FindProtobuf)
find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})
#message("${Protobuf_LIBRARIES}")
#message("${Protobuf_INCLUDE_DIRS}")

# Find the ZeroMQ package
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZMQ REQUIRED libzmq)

set(PROTO_FILES ./message_vec.proto)

# Generate the .pb.h and .pb.cc files
#protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Specify the C++ standard (We use latest version c++20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Specify the folder containing your source files
set(LIBRARY_SOURCE_DIR ${CMAKE_SOURCE_DIR}/library)
file(GLOB LIBRARY_SOURCES ${LIBRARY_SOURCE_DIR}/*.cc
                          ${LIBRARY_SOURCE_DIR}/*.cpp)
file(GLOB LIBRARY_SOURCES_HEADER ${LIBRARY_SOURCE_DIR}/*.h
                          ${LIBRARY_SOURCE_DIR}/*.hpp)

# Include the directory where zmq, .pb.h, and are hpp files 
include_directories(${ZMQ_INCLUDE_DIRS} ${LIBRARY_SOURCE_DIR})

# Get C or C++ sources in the current directory (only).
file(GLOB C_SOURCES *.c)
file(GLOB CPP_SOURCES *.cpp)
set(SOURCES ${C_SOURCES} ${CPP_SOURCES} ${LIBRARY_SOURCES})

# Set the  Webots home path (change it according to your installation method)
set(WEBOTS_HOME "/usr/local/webots")
#set(WEBOTS_HOME "/snap/webots/current/usr/share/webots")

# Link with the Webots controller library.
link_directories($ENV{WEBOTS_HOME}/lib/controller)
set (LIBRARIES m ${CMAKE_SHARED_LIBRARY_PREFIX}Controller${CMAKE_SHARED_LIBRARY_SUFFIX} ${CMAKE_SHARED_LIBRARY_PREFIX}CppController${CMAKE_SHARED_LIBRARY_SUFFIX})
include_directories($ENV{WEBOTS_HOME}/include/controller/c $ENV{WEBOTS_HOME}/include/controller/cpp)

# Find the OpenCV package
find_package(OpenCV REQUIRED)

# Find the PointCloud Library package
# find_package(PCL REQUIRED)

# Include the OpenCV headers
# message(opencv_check " : ${OpenCV_INCLUDE_DIRS}")
include_directories(${OpenCV_INCLUDE_DIRS})

# Include the PointCloud Library headers
# include_directories(${PCL_LIBRARY_DIRS})
# Add Definition for PointCloud Library
# add_definitions(${PCL_DEFINITIONS})

# Setup the target executable.
add_executable(${PROJECT} ${SOURCES} )
################ROS2-ONLY################
ament_target_dependencies(${PROJECT} rclcpp sensor_msgs nav_msgs tf2 tf2_ros tf2_geometry_msgs)
target_link_libraries(${PROJECT} ${LIBRARIES} ${OpenCV_LIBS} #${PCL_LIBRARIES} 
                      ${Protobuf_LIBRARIES} ${ZMQ_LIBRARIES})

# Disable the -Wreorder warning
target_compile_options(${PROJECT} PRIVATE -Wno-reorder -Wpointer-arith)

## Copy the target executable at the right location.
#add_custom_command(TARGET ${PROJECT} POST_BUILD COMMAND ${CMAKE_COMMAND} -E
#  copy ${CMAKE_BINARY_DIR}/${PROJECT} ${CMAKE_SOURCE_DIR}
#)

install(TARGETS
	${PROJECT}
	DESTINATION lib/${PROJECT}
	)
	
install(DIRECTORY config/
	DESTINATION share/${PROJECT_NAME}/config
	)
	
ament_package()
