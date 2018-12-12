#define NO_QT
#include <render/quadRender/quadRender.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <texture_types.h>

__device__ __constant__ SceneInformation cScene; ///< Global accessor for scene information from constant memory in
                                                 ///< cuda

surface<void, cudaSurfaceType2D> surfaceWriteOut; ///< Surface handle used to render from cuda

__global__ void renderfunc(DeviceCamera cam) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= cScene.width)
    return;
  if (y >= cScene.height)
    return;

  surf2Dwrite(float4{x / 1920.f, y / 1080.f, 0.f, 1.f}, surfaceWriteOut, x * sizeof(float4), y, cudaBoundaryModeClamp);
}

void cudaRender(SceneInformation scene, cudaGraphicsResource_t resource) {
  static bool once = true;
  if (once) {
    cudaArray_t color_arr;
    cudaGraphicsMapResources(1, &resource, 0);
    cudaGraphicsSubResourceGetMappedArray(&color_arr, resource, 0, 0);
    cudaBindSurfaceToArray(surfaceWriteOut, color_arr);
    once = false;
  }

  cudaMemcpyToSymbol(cScene, &scene, sizeof(SceneInformation));

  dim3 texture_dim(scene.width, scene.height, 1);
  dim3 block_dim(32, 16, 1);
  dim3 grid_dim(texture_dim.x / block_dim.x, texture_dim.y / block_dim.y, 1);

  if (texture_dim.x % block_dim.x != 0)
    grid_dim.x += 1;
  if (texture_dim.y % block_dim.y != 0)
    grid_dim.y += 1;

  renderfunc<<<grid_dim, block_dim>>>(scene.m_camera);
  cudaDeviceSynchronize();
}