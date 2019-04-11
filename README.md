# dnn_fast_resample
Code for testing faster kernels for bilinear interpolation operator in DNNs

program output on GTX1080:
```
$ ./bilin_test
read image of size 1080x1920
resized to size 200x100
Allocated CHWTensor of size (128 x 200 x 100)
tensor created
tensor written to HDD
Invoking warmup
tensor is now on GPU
warmup complete
tensor is now on GPU
1000 kernel 1 invokations lasted for 46927611mus
1000 kernel 2 invokations lasted for 428558mus
```
