#include <torch/torch.h>
#include <iostream>

int main() {
  std::cout << torch::cuda::is_available() << std::endl;
}
