load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_test")

cc_test(
    name = "main_test",
    srcs = ["main_test.cc"],
    deps = ["@googletest//:gtest", "@googletest//:gtest_main"],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "main",
    srcs = ["main.cc"],
    visibility = ["//visibility:public"],
    deps = ["@libtorch//:torch", "@tensorboard_logger"],
)
