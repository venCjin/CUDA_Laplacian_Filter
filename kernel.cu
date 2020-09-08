#include "kernel.h"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define LAP_3x3 \
{ \
   -1, -1, -1, \
   -1,  8, -1, \
   -1, -1, -1  \
};
#define LAP_5x5 \
{ \
   -1, -3, -4, -3, -1, \
   -3,  0,  6,  0, -3, \
   -4,  6, 20,  6, -4, \
   -3,  0,  6,  0, -3, \
   -1, -3, -4, -3, -1, \
};
#define LAP_7x7 \
{ \
   -2, -3, -4, -6, -4, -3, -2, \
   -3, -5, -4, -3, -4, -5, -3, \
   -4, -4,  9, 20,  9, -4, -4, \
   -6, -3, 20, 36, 20, -3, -6, \
   -4, -4,  9, 20,  9, -4, -4, \
   -3, -5, -4, -3, -4, -5, -3, \
   -2, -3, -4, -6, -4, -3, -2  \
};
const int KERNEL_DIM = 3;
__device__ __constant__ float d_KERNEL[KERNEL_DIM * KERNEL_DIM] = LAP_3x3;

__global__ void laplace (unsigned char* input_image, unsigned char* output_image, int width, int height)
{
    const unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int x = offset % width;
    int y = (offset - x) / width;

    if (offset < width * height)
    {
        float r, g, b;
        r = g = b = 0;

        for (int ox = -(KERNEL_DIM / 2); ox < (KERNEL_DIM / 2) + 1; ox++)
        {
            for (int oy = -(KERNEL_DIM / 2); oy < (KERNEL_DIM / 2) + 1; oy++)
            {
                if ((x + ox) > -1 && (x + ox) < width && (y + oy) > -1 && (y + oy) < height)
                {
                    const int current_offset = (offset + ox + oy * width) * 3;
                    const int kernel_index = ((KERNEL_DIM * KERNEL_DIM) / 2) + ox + (oy * KERNEL_DIM);

                    r += input_image[current_offset]     * d_KERNEL[kernel_index];
                    g += input_image[current_offset + 1] * d_KERNEL[kernel_index];
                    b += input_image[current_offset + 2] * d_KERNEL[kernel_index];
                }
            }
        }
        output_image[offset * 3]     = r;
        output_image[offset * 3 + 1] = g;
        output_image[offset * 3 + 2] = b;
    }
}

void getError (cudaError_t err)
{
    if (err != cudaSuccess)
    {
        std::cout << "Error " << cudaGetErrorString(err) << std::endl;
    }
}

void filter (unsigned char* input_image, unsigned char* output_image, int width, int height)
{
    unsigned char* dev_input;
    unsigned char* dev_output;
    getError(cudaMalloc((void**) &dev_input, width*height*3*sizeof(unsigned char)));
    getError(cudaMemcpy(dev_input, input_image, width*height*3*sizeof(unsigned char), cudaMemcpyHostToDevice));
 
    getError(cudaMalloc((void**) &dev_output, width*height*3*sizeof(unsigned char) ));

    dim3 blockDims(512,1,1);
    dim3 gridDims((unsigned int) ceil((double)(width*height/blockDims.x)), 1, 1);

    laplace<<<gridDims, blockDims>>>(dev_input, dev_output, width, height); 

    getError(cudaMemcpy(output_image, dev_output, width*height*3*sizeof(unsigned char), cudaMemcpyDeviceToHost));

    getError(cudaFree(dev_input));
    getError(cudaFree(dev_output));
}
