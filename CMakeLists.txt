cmake_minimum_required(VERSION 3.10.0)
project(kalman_cpp VERSION 0.1.0 LANGUAGES C CXX)


set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(src)
include_directories(include)

file (GLOB MAIN 
    src/*.cpp
    )

add_executable(kalman_cpp ${MAIN})

include(CTest)
enable_testing()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
