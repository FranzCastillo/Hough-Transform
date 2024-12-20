/*
 ============================================================================
 Author        : G. Barlas
 Version       : 2.0
 Last modified : November 2024
 Modified by   : A. Montoya, F. Esquivel, F. Castillo
 License       : Released under the GNU GPL 3.0
 Description   :
 To build use  : make
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include <fstream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "../../common/pgm.h"

const double degreeInc = 0.5; // use 4.0 for reinforced-lines image and 0.5 for original
const int degreeBins = static_cast<int>(180.0 / degreeInc);
const int rBins = 100;
const double radInc = degreeInc * M_PI / 180.0;

__constant__ double d_CosConst[degreeBins];
__constant__ double d_SinConst[degreeBins];

//*****************************************************************
// The CPU function returns a pointer to the accummulator
void CPU_HoughTran(unsigned char *pic, int w, int h, int **acc)
{
    double rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2.0;
    *acc = new int[rBins * degreeBins];
    memset(*acc, 0, sizeof(int) * rBins * degreeBins);
    int xCent = w / 2;
    int yCent = h / 2;
    double rScale = 2.0 * rMax / rBins;

    for (int i = 0; i < w; i++)
    {
        for (int j = 0; j < h; j++)
        {
            int idx = j * w + i;
            if (pic[idx] > 0)
            {
                int xCoord = i - xCent;
                int yCoord = yCent - j;
                double theta = 0.0;
                for (int tIdx = 0; tIdx < degreeBins; tIdx++)
                {
                    double r = xCoord * cos(theta) + yCoord * sin(theta);
                    int rIdx = (r + rMax) / rScale;
                    if (rIdx >= 0 && rIdx < rBins)
                    {
                        (*acc)[rIdx * degreeBins + tIdx]++;
                    }
                    theta += radInc;
                }
            }
        }
    }
}

// GPU kernel. One thread per image pixel is spawned.
// The accummulator memory needs to be allocated by the host in global memory
__global__ void GPU_HoughTran(unsigned char *pic, int w, int h, int *acc,
    double rMax, double rScale, double *d_Cos, double *d_Sin)
{
  int gloID = blockIdx.x * blockDim.x + threadIdx.x;
  if (gloID >= w * h)
    return;

  int xCent = w / 2;
  int yCent = h / 2;
  int xCoord = gloID % w - xCent;
  int yCoord = yCent - gloID / w;

  if (pic[gloID] > 0)
  {
    for (int tIdx = 0; tIdx < degreeBins; tIdx++)
    {
        double r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
        int rIdx = (r + rMax) / rScale;
        if (rIdx >= 0 && rIdx < rBins)
        {
            atomicAdd(acc + (rIdx * degreeBins + tIdx), 1);
        }
    }
  }
}

//*****************************************************************
// Constant memory kernel
__global__ void GPU_HoughTranConst(unsigned char *pic, int w, int h, int *acc,
    double rMax, double rScale)
{
    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    if (gloID >= w * h)
        return;

    int xCent = w / 2;
    int yCent = h / 2;
    int xCoord = gloID % w - xCent;
    int yCoord = yCent - gloID / w;

    if (pic[gloID] > 0)
    {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++)
        {
            double r = xCoord * d_CosConst[tIdx] + yCoord * d_SinConst[tIdx];
            int rIdx = (r + rMax) / rScale;
            if (rIdx >= 0 && rIdx < rBins)
            {
                atomicAdd(acc + (rIdx * degreeBins + tIdx), 1);
            }
        }
    }
}

//*****************************************************************
//Shared Memory Kernel
__global__ void GPU_HoughTranSharedFixed(unsigned char *pic, int w, int h, int *acc,
                                         double rMax, double rScale) {
    //Global and Local IDs
    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    int locID = threadIdx.x;

    if (gloID >= w * h) return;

    //Fragment size on shared memory
    const int fragmentSize = 256;
    extern __shared__ int localAcc[]; //Dynamic shared memory allocation

    //Inicialize the local accumulator
    for (int i = locID; i < fragmentSize; i += blockDim.x) {
        localAcc[i] = 0;
    }
    __syncthreads(); //Sincronization for initialization

    //Image center
    int xCent = w / 2;
    int yCent = h / 2;

    //Relative to the center coordinates
    int xCoord = gloID % w - xCent;
    int yCoord = yCent - gloID / w;

    //Proccess only white pixels
    if (pic[gloID] > 0) {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
            double r = xCoord * d_CosConst[tIdx] + yCoord * d_SinConst[tIdx];
            int rIdx = (r + rMax) / rScale;

            if (rIdx >= 0 && rIdx < rBins) {
                //Map global index to local index
                int globalIdx = rIdx * degreeBins + tIdx;
                int localIdx = globalIdx % fragmentSize;
                atomicAdd(&localAcc[localIdx], 1); //Increment the local accumulator
            }
        }
    }
    __syncthreads(); //Sincronize before transferring results

    //Transfer local accumulator to global accumulator
    for (int i = locID; i < fragmentSize; i += blockDim.x) {
        int globalIdx = blockIdx.x * fragmentSize + i; //Map to global accumulator
        if (globalIdx < degreeBins * rBins) {
            atomicAdd(&acc[globalIdx], localAcc[i]);
        }
    }
}

//*****************************************************************
int main(int argc, char **argv)
{
  int i;

  PGMImage inImg(argv[1]);

  int *cpuht;
  int w = inImg.x_dim;
  int h = inImg.y_dim;
  
  double* d_Cos;
  double* d_Sin;

  cudaMalloc ((void **) &d_Cos, sizeof (double) * degreeBins);
  cudaMalloc ((void **) &d_Sin, sizeof (double) * degreeBins);

  // CPU calculation
  CPU_HoughTran(inImg.pixels, w, h, &cpuht);

  // pre-compute values to be stored
  double *pcCos = (double *)malloc(sizeof(double) * degreeBins);
  double *pcSin = (double *)malloc(sizeof(double) * degreeBins);
  double rad = 0.0;
  for (int i = 0; i < degreeBins; i++)
  {
      pcCos[i] = cos(rad);
      pcSin[i] = sin(rad);
      rad += radInc;
  }

  cudaMemcpy(d_Cos, pcCos, sizeof(double) * degreeBins, cudaMemcpyHostToDevice);
  cudaMemcpy(d_Sin, pcSin, sizeof(double) * degreeBins, cudaMemcpyHostToDevice);

  cudaMemcpyToSymbol(d_CosConst, pcCos, sizeof(double) * degreeBins);
  cudaMemcpyToSymbol(d_SinConst, pcSin, sizeof(double) * degreeBins);


  double rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
  double rScale = 2 * rMax / rBins;

  // setup and copy data from host to device
  unsigned char *d_in, *h_in;
  int *d_hough, *h_hough;

  h_in = inImg.pixels; // h_in had the image pixels

  h_hough = (int *)malloc(degreeBins * rBins * sizeof(int));

  cudaMalloc((void **)&d_in, sizeof(unsigned char) * w * h);
  cudaMalloc((void **)&d_hough, sizeof(int) * degreeBins * rBins);
  cudaMemcpy(d_in, h_in, sizeof(unsigned char) * w * h, cudaMemcpyHostToDevice);
  cudaMemset(d_hough, 0, sizeof(int) * degreeBins * rBins);

  // execution configuration uses a 1-D grid of 1-D blocks, each made of 256 threads
  // 1 thread por pixel
  int blockNum = ceil(w * h / 256);

  //Create CUDA events to measure time
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  //Execute kernel with constant memory
  cudaEventRecord(start);
  GPU_HoughTranConst<<<blockNum, 256>>>(d_in, w, h, d_hough, rMax, rScale);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float millisecondsConst = 0;
  cudaEventElapsedTime(&millisecondsConst, start, stop);
  printf("Tiempo de ejecución del kernel (memoria constante): %f ms\n", millisecondsConst);

  // Clear accumulator and execute kernel with global memory
  cudaMemset(d_hough, 0, sizeof(int) * degreeBins * rBins);
  cudaEventRecord(start);
  GPU_HoughTran<<<blockNum, 256>>>(d_in, w, h, d_hough, rMax, rScale, d_Cos, d_Sin);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float millisecondsGlobal = 0;
  cudaEventElapsedTime(&millisecondsGlobal, start, stop);
  printf("Tiempo de ejecución del kernel (memoria global): %f ms\n", millisecondsGlobal);

  //Calculate the required shared memory size
  int sharedMemorySize = degreeBins * sizeof(int);
  //Execute kernel with shared memory
  cudaEventRecord(start);
  GPU_HoughTranSharedFixed<<<blockNum, 256, sharedMemorySize>>>(d_in, w, h, d_hough, rMax, rScale);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float millisecondsShared = 0;
  cudaEventElapsedTime(&millisecondsShared, start, stop);
  printf("Tiempo de ejecución del kernel (memoria compartida fija): %f ms\n", millisecondsShared);


  // get results from device
  cudaMemcpy(h_hough, d_hough, sizeof(int) * degreeBins * rBins, cudaMemcpyDeviceToHost);
  
  printf("Done!\n");

  cv::Mat originalImage = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
  if (originalImage.empty())
  {
      std::cerr << "Error: Could not load image " << argv[1] << std::endl;
      return -1;
  }

  // Convert to BGR color space to draw colored lines
  cv::Mat colorImage;
  cv::cvtColor(originalImage, colorImage, cv::COLOR_GRAY2BGR);

  // Set threshold for significant lines

  const int threshold = 4200; //Use 2500 for reinforced-lines image an 4200 for original lines

  // Apply threshold
  for (int rIdx = 0; rIdx < rBins; rIdx++)
    {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++)
        {
            int idx = rIdx * degreeBins + tIdx;
            if (h_hough[idx] >= threshold)
            {
                float theta = tIdx * radInc;
                float r = rIdx * rScale - rMax;

                double cosTheta = cos(theta);
                double sinTheta = sin(theta);

                // Points where the line crosses the borders of the image
                cv::Point pt1, pt2;

                // Since sinTheta and cosTheta can be zero, we need to handle those cases
                if (fabs(sinTheta) > 1e-6)
                {
                    // Compute the intersection with the left and right borders
                    pt1.x = 0;
                    pt1.y = (r - (pt1.x - w / 2) * cosTheta) / sinTheta + h / 2;

                    pt2.x = w;
                    pt2.y = (r - (pt2.x - w / 2) * cosTheta) / sinTheta + h / 2;
                }
                else
                {
                    // sinTheta is zero, line is horizontal
                    pt1.y = 0;
                    pt1.x = (r - (pt1.y - h / 2) * sinTheta) / cosTheta + w / 2;

                    pt2.y = h;
                    pt2.x = (r - (pt2.y - h / 2) * sinTheta) / cosTheta + w / 2;
                }

                // Adjust y-coordinates to account for the inverted y-axis
                pt1.y = h - pt1.y;
                pt2.y = h - pt2.y;

                // Draw the line on the image
                cv::line(colorImage, pt1, pt2, cv::Scalar(0, 255, 0), 1);
            }
        }
    }

  cv::imwrite("output.png", colorImage);

  // cleanup
  free(cpuht);
  free(h_hough);
  free(pcCos);
  free(pcSin);
  cudaFree(d_in);
  cudaFree(d_hough);
  cudaFree(d_Cos);
  cudaFree(d_Sin);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
