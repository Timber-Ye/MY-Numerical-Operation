cmake_minimum_required(VERSION 3.19)
project(OperationResearch)

set(CMAKE_CXX_STANDARD 17)

include_directories(.)
include_directories(D:/Installation/eigen)
set(SOURCE_FILES Src/main.cpp Inc/numOpt.h Src/Unconstrained_opt.cpp Inc/Unconstrained_opt.h Inc/OptCore.h Src/Heuristic_opt.cpp Inc/Heuristic_opt.h Inc/heuOpt.h)

add_executable(OperationResearch ${SOURCE_FILES})
