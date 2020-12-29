#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "image_blur.h"
#include "helpers.h"
#include <iostream>
#include <cmath>

__global__ void blur_cuda(unsigned char* input_image, unsigned char* output_image, int width, int height);



void image_blur_cuda(unsigned char* Input_Image, int Height, int Width, int Channels) {
	unsigned char* Dev_Input_Image = NULL;
	unsigned char* Dev_Output_Image = NULL;



	getError(cudaMalloc((void**)&Dev_Input_Image, Width * Height * 3 * sizeof(unsigned char)));
	getError(cudaMemcpy(Dev_Input_Image, Input_Image, Width * Height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice));

	getError(cudaMalloc((void**)&Dev_Output_Image, Width * Height * 3 * sizeof(unsigned char)));
    
    
    dim3 blockDims(512, 1, 1);
	dim3 gridDims((unsigned int)ceil((double)(Width * Height * 3 / blockDims.x)), 1, 1);
	

	

	blur_cuda << <gridDims, blockDims >> > (Dev_Input_Image, Dev_Output_Image,Width,Height);

	//copy processed data back to cpu from gpu
	getError(cudaMemcpy(Input_Image, Dev_Output_Image, Width * Height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost));

	getError(cudaFree(Dev_Input_Image));
	getError(cudaFree(Dev_Output_Image));
	//free gpu mempry
	
}


__global__ void blur_cuda(unsigned char* input_image, unsigned char* output_image, int Width, int Height) {
    const unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int x = offset % Width;
    int y = (offset - x) / Width;
    int fsize = 5; // Filter size
    if (offset < Width * Height) {

        float output_red = 0;
        float output_green = 0;
        float output_blue = 0;
        int hits = 0;
        for (int ox = -fsize; ox < fsize + 1; ++ox) {
            for (int oy = -fsize; oy < fsize + 1; ++oy) {
                if ((x + ox) > -1 && (x + ox) < Width && (y + oy) > -1 && (y + oy) < Height) {
                    const int currentoffset = (offset + ox + oy * Width) * 3;
                    output_red += input_image[currentoffset];
                    output_green += input_image[currentoffset + 1];
                    output_blue += input_image[currentoffset + 2];
                    hits++;
                }
            }
        }
        output_image[offset * 3] = output_red / hits;
        output_image[offset * 3 + 1] = output_green / hits;
        output_image[offset * 3 + 2] = output_blue / hits;
    }
}