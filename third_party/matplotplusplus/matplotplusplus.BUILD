"""Matplot++ extention for bazel"""

load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])
licenses(["notice"])
exports_files(["LICENSE"])

cc_library(
    name = "cimg",
    hdrs = ["source/3rd_party/cimg/CImg.h"],
    copts = ["-std=c++17","-w"],
    strip_include_prefix = "source/3rd_party/cimg",
)

cc_library(
    name = "nodesoup",
    srcs = glob([
        "source/3rd_party/nodesoup/**/*.cpp",
        "source/3rd_party/nodesoup/**/*.hpp",
    ]),
    hdrs = ["source/3rd_party/nodesoup/include/nodesoup.hpp"],
    copts = ["-std=c++17","-w"],
    strip_include_prefix = "source/3rd_party/nodesoup/include/",
)

cc_library(
    name = "matplot",
    srcs = glob(
        include = ["source/matplot/**/*.cpp"],
        exclude = ["**/*opengl*"],
    ),
    hdrs = glob(
        include = ["source/matplot/**/*.h" ],
        exclude = ["**/*opengl*"],
    ),
    copts = ["-std=c++17","-w"],
    linkopts = ["-lX11"],
    strip_include_prefix = "source",
    deps = [":cimg",":nodesoup"],
)
