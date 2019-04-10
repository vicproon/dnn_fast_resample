#include "call_kernel.h"
#include "input_data.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

CHWTensor generate_input_tensor()
{
  // read input image as HWC 
  cv::Mat img = cv::imread("/home/prun/Y/box/prun/inference_input/pgr.163.002.000050.png");
  // resize it and duplicate rgb channels so we have 128x20x10 tensor
  cv::Mat small_img; cv::resize(img, small_img, cv::Size(100, 200));
  std::cout << "read image of size " << img.rows << "x" << img.cols << "\n";
  std::cout << "resized to size " << small_img.rows << "x" << small_img.cols << "\n";
  
  cv::Mat small_img_f32; small_img.convertTo(small_img_f32, CV_32FC1);
  std::vector<cv::Mat> channels;
  cv::split(small_img_f32, channels);

  if(!channels[0].isContinuous())
    throw std::logic_error("channel mat is not continous!\n");

  CHWTensor t = CHWTensor::zeros(128, small_img.rows, small_img.cols);
  std::cout << "Allocated CHWTensor of size (" << t.channels << " x " << t.height << " x " <<
               t.width << ")\n";
  for (int i = 0; i < t.channels; ++i)
  {
    const auto &ch = channels[i % channels.size()];
    float *p = t.data_begin + i * t.width * t.height;
    memcpy(p, reinterpret_cast<void *> (ch.data), sizeof(float) * t.width * t.height);
  }

  return t;
}

