project(unitree_slam_demo)
cmake_minimum_required(VERSION 3.16)

include_directories(/usr/local/include/ddscxx /usr/local/include/iceoryx/v2.0.2 /usr/local/include/unitree)
link_libraries(unitree_sdk2 unitree_ros2_idl_cpp ddsc ddscxx rt pthread)

add_executable(demo demo.cpp)


