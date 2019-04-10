#include "call_kernel.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>

typedef float Data;

__global__
void resize_bilinear_kernel_2d(int nbatch,
                              int ichan,
                              int iheight,
                              int iwidth,
                              int2 osize,
                              Data const* idata,
                              Data*       odata) {
  int index_x = threadIdx.x + blockIdx.x * blockDim.x;
  int index_y = threadIdx.y + blockIdx.y * blockDim.y;
  int index = index_y * osize.x + index_x;

  int channels = ichan;
  int width1 = iwidth;
  int width2 = osize.x;
  int height1 = iheight;
  int height2 = osize.y;

  const int n = height2 * width2;
  
  if (index >= n)
    return;
  const int n_in = height1 * width1;
    const int w2 = index % width2; // 0:width2-1 
    const int h2 = index / width2; // 0:height2-1
    const float rheight =(height2 > 1) ? static_cast<float>(height1- 1)/
                                                (height2 - 1) : 0.f;
    const float rwidth = (width2 > 1) ? static_cast<float>(width1- 1) /
                                                (width2 - 1) : 0.f;
    const float h1r = rheight * h2;
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const int h1ps = h1p * width1;
    const Data h1lambda = h1r - h1;
    const Data h0lambda = Data(1.) - h1lambda;
    //
    const float w1r = rwidth * w2;
    const int w1 = w1r;
    const int w1p = (w1 < width1 - 1) ? 1 : 0;
    const Data w1lambda = w1r - w1;
    const Data w0lambda = Data(1.) - w1lambda;
    //

    const Data* pos1_tl = &idata[h1 * width1 + w1];
    const Data* pos1_tr = pos1_tl + w1p;
    const Data* pos1_bl = pos1_tl + h1ps;
    const Data* pos1_br = pos1_bl + w1p;

    const Data h0w0 = h0lambda * w0lambda;
    const Data h0w1 = h0lambda * w1lambda;
    const Data h1w0 = h1lambda * w0lambda;
    const Data h1w1 = h1lambda * w1lambda;

    Data* pos2 = &odata[h2 * width2 + w2];
    for (int c = 0; c < channels; ++c) {
        pos2[0] = 
                h0w0 * *pos1_tl + h0w1 * *pos1_tr +
                h1w0 * *pos1_bl + h1w1 * *pos1_br;
        pos1_tl += n_in;
        pos1_tr += n_in;
        pos1_bl += n_in;
        pos1_br += n_in;
        pos2 += n;
    }
}


__global__
void resize_bilinear_kernel_2d_c(int nbatch,
                              int ichan,
                              int iheight,
                              int iwidth,
                              int2 osize,
                              Data const* idata,
                              Data*       odata) {
                                  
  const int index_x = threadIdx.x + blockIdx.x * blockDim.x;
  const int index_c = threadIdx.y + blockIdx.y * blockDim.y;
  const int index_y = threadIdx.z + blockIdx.z * blockDim.z;
  
 // printf("(%d, %d, %d);", index_x, index_y, index_c);

  const int channels = ichan;
  const int width1 = iwidth;
  const int width2 = osize.x;
  const int height1 = iheight;
  const int height2 = osize.y;
  
  const int n = height2 * width2;
  const int n_in = height1 * width1;
  
  // int g_index = n * index_c + index_y * osize.x + index_x;
  int index = index_y * osize.x + index_x;
  
  if (index >= n || index_c >= ichan)
    return;
    
    const int w2 = index % width2; // 0:width2-1 
    const int h2 = index / width2; // 0:height2-1
    const float rheight =(height2 > 1) ? static_cast<float>(height1- 1)/
                                                (height2 - 1) : 0.f;
    const float rwidth = (width2 > 1) ? static_cast<float>(width1- 1) /
                                                (width2 - 1) : 0.f;
    const float h1r = rheight * h2;
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const int h1ps = h1p * width1;
    const Data h1lambda = h1r - h1;
    const Data h0lambda = Data(1.) - h1lambda;
    //
    const float w1r = rwidth * w2;
    const int w1 = w1r;
    const int w1p = (w1 < width1 - 1) ? 1 : 0;
    const Data w1lambda = w1r - w1;
    const Data w0lambda = Data(1.) - w1lambda;
    //
    const int c = index_c;

    const Data* pos1_tl = &idata[c * n_in + h1 * width1 + w1];
    const Data* pos1_tr = pos1_tl + w1p;
    const Data* pos1_bl = pos1_tl + h1ps;
    const Data* pos1_br = pos1_bl + w1p;

    const Data h0w0 = h0lambda * w0lambda;
    const Data h0w1 = h0lambda * w1lambda;
    const Data h1w0 = h1lambda * w0lambda;
    const Data h1w1 = h1lambda * w1lambda;
    Data* pos2 = &odata[c * n + h2 * width2 + w2];
    pos2[0] = h0w0 * *pos1_tl + h0w1 * *pos1_tr +
              h1w0 * *pos1_bl + h1w1 * *pos1_br;
    //*pos2 = 128.0;
}

CHWTensor call_kernel_1(int ntimes, CHWTensor *pT)
{
  CHWTensor t = pT->toGPU();
  std::cout << "tensor is now on GPU\n";
  CHWTensor dst = CHWTensor::zeros(t.channels, static_cast<int>(1.65 * t.height),
                                               static_cast<int>(1.9 * t.width)).toGPU();
  
  int batchSize = 1;
  int nchan = t.channels;
  for (int i = 0; i < ntimes; ++i)
  {
    int2 osize{dst.width, dst.height};
    dim3 block(32, 32);
    dim3 grid((osize.x - 1) / block.x + 1,
              (osize.y - 1) / block.y + 1,
              std::min(batchSize * nchan, 65535));
    resize_bilinear_kernel_2d<<<grid, block>>>(1, t.channels, t.height, t.width, osize,
                                 t.data_begin, dst.data_begin); 
  }
  return dst.toCPU();
}

CHWTensor call_kernel_2(int ntimes, CHWTensor *pT)
{
  CHWTensor t = pT->toGPU();
  CHWTensor dst = CHWTensor::zeros(t.channels, static_cast<int>(1.65 * t.height),
                                               static_cast<int>(1.9 * t.width)).toGPU();
  
  int batchSize = 1;
  int nchan = t.channels;
  for (int i = 0; i < ntimes; ++i)
  {
    int2 osize{dst.width, dst.height};
    // for some unknown reason the kernel did not launch with block.z == 128 and block.y == 1.
    // i had to swap them here and their meaning in kernel. The CUDA PROGRAMMING GUIDE sais 
    // something unsupported for Z grid dim only for cards with compute capability 1.X
    // (Appendix B.18)
    dim3 block(8, 128, 1);
    dim3 grid((osize.x - 1) / block.x + 1,
              (std::min(batchSize * nchan, 65535) - 1) / block.y + 1,
              (osize.y - 1) / block.z + 1);
    //printf("grid dims are (%d, %d, %d)\n", grid.x, grid.y, grid.z);
    resize_bilinear_kernel_2d_c<<<grid, block>>>(1, t.channels, t.height, t.width, osize,
                                 t.data_begin, dst.data_begin); 
  }

  return dst.toCPU();
}
