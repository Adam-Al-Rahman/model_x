// Copyright 2025 MODEL_X

#include <iostream>
#include <string>

#include "src/components/tokenizers/tokenizers.h"
#include "torch/torch.h"

int main(int argc, char **argv) {
  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) {
    std::cout << "Device: CUDA" << std::endl;
    device = torch::Device(torch::kCUDA);
  } else {
    std::cout << "Device: CPU" << std::endl;
  }

#ifdef __APPLE__
  device = torch::Device(torch::kMPS);
#endif

  const std::u8string text =
      u8"Trapped-ion research at QSA is rapidly advancing scalable, stable "
      "quantum computing 🧬⚡, with the Enchilada Trap 🫔 designed at Sandia "
      "National Labs enabling storage of up to 200 ions 🧪, reducing RF power "
      "dissipation 🔋⚡, and connecting multiple operational zones 🔗; "
      "parallel gate operations 🚀🧩 at University of Maryland allow "
      "simultaneous control of qubits along different vibrational modes 🎵✨ "
      "for higher throughput ⏱️💨; multi-ion entanglement using squeezing 🌌🔗 "
      "by Duke University efficiently entangles many qubits at once, expanding "
      "entangling gate capabilities 🔥🧠; mid-circuit measurements "
      "🕵️‍♂️🔬 "
      "spatially separate ions to allow classical verification of quantum "
      "advantage 🖥️🪐🔑 via Learning With Errors (LWE) and Computational Bell "
      "Test protocols 🎯🔐; together, these innovations push quantum computing "
      "toward solving previously intractable problems 💡🌟, making systems "
      "more efficient, reliable, interactive, and powerful 🚀💻✨.";

  auto text_encoded =
      model_x::src::component::tokenizers(text, "byte_pair", 500);
  std::cout << text_encoded << std::endl;
}
