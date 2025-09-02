"""Provides the repository macro to import Matplot++."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
def repo():
    """Imports Matplot++ for Bazel."""

    MATPLOT_PP_SHA256 = "0c8d8558a9624f676845f61e39de19b77795cc1a299c43a61cea7e301272c49e"

    http_archive(
        name = "matplot_pp",
        build_file = "//third_party/matplotplusplus:matplotplusplus.BUILD",
        url = "https://github.com/alandefreitas/matplotplusplus/archive/refs/tags/v1.2.2.zip",
        strip_prefix = "matplotplusplus-1.2.2",
        sha256 = MATPLOT_PP_SHA256,
    )

def _matplot_pp_impl(_ctx):
    repo()

matplot_pp_extention = module_extension(
    implementation = _matplot_pp_impl,
)
