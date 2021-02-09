#pragma once
#ifndef HG__INPUT_DATA_H_20190410
#define HG__INPUT_DATA_H_20190410 
#include <cstring>
#include <memory>
#include <functional>

#include <cuda_runtime_api.h>
#include <cuda.h>

#include <iostream>

enum class Ctx
{
  CPU, GPU
};

struct CHWTensor
{
  CHWTensor() = default;

  static CHWTensor zeros(int channels, int height, int width)
  {
    CHWTensor t;
    t.channels = channels;
    t.width = width;
    t.height = height;
    t.data_begin = new float[channels * width * height];
    memset(t.data_begin, 0, sizeof(float) * channels * width * height);
    return t;
  }

  CHWTensor toGPU()
  {
    CHWTensor t;
    t.channels = channels;
    t.width = width;
    t.height = height;
    cudaMalloc(reinterpret_cast<void **> (&t.data_begin), sizeof(float) * channels * width * height);
    cudaMemcpyKind cpyTo = (ctx == Ctx::GPU ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice);
    cudaMemcpy(t.data_begin, data_begin, sizeof(float) * channels * width * height,
               cpyTo);
    t.ctx = Ctx::GPU;

    t.deleter = [](float * p){cudaFree(p);};
    return t;
  }

  CHWTensor toCPU()
  {
    CHWTensor t;
    t.channels = channels;
    t.width = width;
    t.height = height;
    t.data_begin = new float[channels * width * height];
    
    switch (ctx)
    {
    case Ctx::CPU: memcpy(t.data_begin, data_begin, sizeof(float) * channels * width * height);
                   break;
    case Ctx::GPU: cudaMemcpy(t.data_begin, data_begin, sizeof(float) * channels * width * height,
                              cudaMemcpyDeviceToHost);
                   break;
    default:       break;
    }
    
    t.ctx = Ctx::CPU;
    return t;
  } 
  
  CHWTensor(const CHWTensor &) = delete;
  CHWTensor & operator=(const CHWTensor &) = delete;

  CHWTensor(CHWTensor && rhs)
  {
    std::cout << "CHWTensor(CHWTensor &&)\n";;
    if (this != &rhs)
    {
      delete[] data_begin;
      channels = rhs.channels;
      width = rhs.width;
      height = rhs.height;
      data_begin = rhs.data_begin;

      rhs.data_begin = nullptr;
      rhs.channels = 0;
      rhs.width = 0;
      rhs.height = 0;
    }
  }

  CHWTensor & operator=(CHWTensor && rhs)
  {
    std::cout << "CHWTensor& operator=(CHWTensor &&)\n";;
    if (this != &rhs)
    {
      delete[] data_begin;
      channels = rhs.channels;
      width = rhs.width;
      height = rhs.height;
      data_begin = rhs.data_begin;

      rhs.data_begin = nullptr;
      rhs.channels = 0;
      rhs.width = 0;
      rhs.height = 0;
    }
    return *this;
  }

  ~CHWTensor()
  {
    // delete[] data_begin;
    deleter(data_begin);
  }

  float * channelPtr(int ic) const
  {
    return data_begin + ic * width * height;
  }

  float *data_begin = nullptr;
  int channels = 0;
  int width = 0;
  int height = 0;
  Ctx ctx = Ctx::CPU;
  // typedef decltype([](float *)-> void {}) TDeleterLambda;
  typedef std::function<void(float *)> TDeleterLambda;
  TDeleterLambda deleter = [](float * p) {delete[] p;};
};

CHWTensor generate_input_tensor();

#endif //HG__INPUT_DATA_H_20190410 
