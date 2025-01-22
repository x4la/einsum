cmake_minimum_required(VERSION 3.29)
project(untitled)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -fPIC -std=c++14 -fopenmp -march=native")

find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 20)
add_library(bmm_v1 SHARED bmm_v1.cpp)
add_library(bmm_v2 SHARED bmm_v2.cpp)
add_library(bmm_v3 SHARED bmm_v3.cpp)
add_library(bmm_v4 SHARED bmm_v4.cpp kernel.cpp)
add_library(bmm_v5 SHARED trash_bmm_v5.cpp kernel.cpp)
add_library(bmm_v6 SHARED bmm_v6.cpp kernel.cpp)
add_library(bmm_v7 SHARED bmm_v7.cpp kernel.cpp)
add_library(bmm_v8 SHARED bmm_v8.cpp kernel.cpp)
#add_library(mm_col_maj SHARED mm_col_maj.cpp kernel_col_maj.cpp)
#add_library(mm_row_maj SHARED mm_row_maj.cpp kernel_col_maj.cpp)


