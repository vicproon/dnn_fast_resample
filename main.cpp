#include <iostream>
#include <chrono>
#include "call_kernel.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace std::chrono;


void output_tensor(const std::string &suffix, const CHWTensor &t)
{
  for (int i = 0; i < t.channels; ++i)
  {
    cv::Mat c32f(t.height, t.width, CV_32FC1, t.channelPtr(i));
    cv::Mat c8u; c32f.convertTo(c8u, CV_8UC1);
    cv::imwrite(string("../res/") + cv::format("%03d_", i) + suffix + ".ppm", c8u);
  }
}

void output_tensor_gt(const std::string &suffix, const CHWTensor &t, cv::Size dst_size)
{
  for (int i = 0; i < t.channels; ++i)
  {
    cv::Mat c32f_small(t.height, t.width, CV_32FC1, t.channelPtr(i));
    cv::Mat c32f; cv::resize(c32f_small, c32f, dst_size);
    cv::Mat c8u; c32f.convertTo(c8u, CV_8UC1);
    cv::imwrite(string("../res/") + cv::format("%03d_", i) + suffix + ".ppm", c8u);
  }
}

int main()
{
  // get input dat
  auto t = generate_input_tensor();
  std::cout << "tensor created\n";
  output_tensor("src", t);
  std::cout << "tensor written to HDD\nInvoking warmup\n";
  //  warmup
  call_kernel_1(10, &t);
  call_kernel_2(10, &t);
  std::cout << "warmup complete\n";
  // test
  {
    auto start = system_clock::now();
    auto dst = call_kernel_1(100, &t);
    auto end = system_clock::now();
    cout << "100 kernel 1 invokations lasted for " << duration_cast<microseconds> (end - start).count()
         << "mus\n";
    output_tensor("dst_ker_1", dst);
    output_tensor_gt("gt", t, {dst.width, dst.height});
  }

  {
    auto start = system_clock::now();
    auto dst = call_kernel_2(100, &t);
    auto end = system_clock::now();
    cout << "100 kernel 2 invokations lasted for " << duration_cast<microseconds> (end - start).count()
         << "mus\n";
    output_tensor("dst_ker_2", dst);
  } 
  return 0;
}
