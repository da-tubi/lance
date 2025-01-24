#  Copyright 2022 Lance Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

cmake_minimum_required(VERSION 3.22)

project(lance CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Default to Debug build if not specified
if (NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE Debug)
endif ()

add_compile_options(-Wall -Wextra -fPIC)

# TODO: Fix on linux
set(OPENSSL_ROOT_DIR /opt/homebrew/opt/openssl@1.1/)

Include(FetchContent)

# TODO: We can remove this package once std::format is available.
FetchContent_Declare(fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG 8.1.1
        )
list(APPEND available_contents fmt)

if (NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    FetchContent_Declare(
            Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2.git
            GIT_TAG v3.0.1
    )
    list(APPEND available_contents catch2)
endif ()
FetchContent_MakeAvailable(${available_contents})

include_directories(${fmt_SOURCE_DIR}/include)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    list(APPEND CMAKE_MODULE_PATH /usr/lib/x86_64-linux-gnu/cmake/arrow)
endif ()

find_package(Arrow REQUIRED)
if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    find_package(Zstd REQUIRED)
    list(APPEND CMAKE_MODULE_PATH /opt/homebrew/opt/apache-arrow/lib/cmake/arrow)
endif ()

find_package(ArrowDataset REQUIRED)
include_directories(include)
include_directories(src)
include_directories(${CMAKE_BINARY_DIR}/src)  # for format.pb.{h/cc}

function(add_lance_test test_name)
    if (NOT CMAKE_BUILD_TYPE STREQUAL "Release")
        add_executable(${test_name} ${test_name}.cc)
        target_link_libraries(${test_name}
                Catch2::Catch2WithMain
                lance
                )
        target_include_directories(${test_name} SYSTEM PRIVATE ${ARROW_INCLUDE_DIR})
        target_include_directories(${test_name} SYSTEM PRIVATE ${PARQUET_INCLUDE_DIR})
        add_test(NAME ${test_name} COMMAND ${test_name})
    endif ()
endfunction()

if (NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    include(CTest)
    include(Catch)
    set(test_libs Catch2::Catch2WithMain)
    enable_testing()
endif ()

set(lance_objects
        $<TARGET_OBJECTS:arrow>
        $<TARGET_OBJECTS:encodings>
        $<TARGET_OBJECTS:format>
        $<TARGET_OBJECTS:io>
        )

add_subdirectory(src)

add_library(lance SHARED ${lance_objects})
target_link_libraries(lance PUBLIC
        ${ARROW_SHARED_LIB}
        ${ARROW_DATASET_SHARED_LIB}
        fmt::fmt
        ${Protobuf_LIBRARIES}
        )
