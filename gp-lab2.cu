#include <iostream>
#include <algorithm>
#include <string>
#include <cstdio>
#include <cstdlib>
//#include "../lib/cuPrintf.cu"

using namespace std;

/*
=========
CONSTANTS
=========
*/

const uint32_t COUNTING_SORT_BASE = 256;
const uint32_t BLOCK_DIM = 32;

namespace Pixel {
	//const uint32_t ELEMENTS_CNT = 4;
	const uint32_t RED = 8 * 0;
	const uint32_t GREEN = 8 * 1;
	const uint32_t BLUE = 8 * 2;
	const uint32_t ALPHA = 8 * 3;
}

/*
==========
STRUCTURES
==========
*/

/*struct Pixel {
	uint8_t Red;
	uint8_t Green;
	uint8_t Blue;
	uint8_t Alpha;
};*/

struct Position {
	int32_t X;
	int32_t Y;
};

/*
======
DEVICE
======
*/

/*__device__ double GetIntensity(Pixel pixel) {
	return (.3 * (double) pixel.Red) + (.59 * (double) pixel.Green) + (.11 * (double) pixel.Blue);
}*/

__device__ bool IsCorrectPos(Position pos, uint32_t height, uint32_t width) {
	if (pos.X >= 0 && pos.Y >= 0 && pos.X < (int32_t) width && pos.Y < (int32_t) height) {
		return true;
	}
	return false;
}

__device__ int32_t GetLinearizedPosition(Position pos, uint32_t height, uint32_t width) {
	if (!IsCorrectPos(pos, height, width)) {
		return -1;
	}
	return pos.Y * width + pos.X;
}

__device__ uint8_t GetMedianElementFromCountArray(uint32_t *arr, uint32_t size) {
	uint32_t tmp_cnt = 0;
	uint8_t res = 0;
	for (uint32_t i = 0; i < COUNTING_SORT_BASE; i++) {
		tmp_cnt += arr[i];
		if (tmp_cnt > size / 2) {
			res = (uint8_t) i;
			break;
		}
	}

	return res;
}

__device__ uint32_t MakePixel(uint8_t red, uint8_t green, uint8_t blue, uint8_t alpha) {
	return ((uint32_t) red << Pixel::RED) + ((uint32_t) green << Pixel::GREEN) +
			((uint32_t) blue << Pixel::BLUE) + ((uint32_t) alpha << Pixel::ALPHA);
}

__device__ uint8_t GetPixelElement(uint32_t pixel, uint32_t element) {
	return (pixel >> element) & 255;
}

__device__ uint32_t GetMedianValue(uint32_t *map_in, uint32_t height, uint32_t width, Position start, Position end) {
	uint32_t count_array_red[COUNTING_SORT_BASE];
	uint32_t count_array_green[COUNTING_SORT_BASE];
	uint32_t count_array_blue[COUNTING_SORT_BASE];
	for (uint32_t i = 0; i < COUNTING_SORT_BASE; i++) {
		count_array_red[i] = 0;
		count_array_green[i] = 0;
		count_array_blue[i] = 0;
	}

	Position curr_pos;

	uint32_t size = 0;
	//cuPrintf("\nStart: [%d:%d]\nEnd: [%d:%d]\n\n", start.X, start.Y, end.X, end.Y);

	for (curr_pos.X = start.X; curr_pos.X <= end.X; curr_pos.X++) {
		for (curr_pos.Y = start.Y; curr_pos.Y <= end.Y; curr_pos.Y++) {
			if (!IsCorrectPos(curr_pos, height, width)) {
				//cuPrintf(" [%d:%d] - INCORRECT\n", curr_pos.X, curr_pos.Y);
				continue;
			}
			//cuPrintf(" [%d:%d] - CORRECT\n", curr_pos.X, curr_pos.Y);
			count_array_red[GetPixelElement(map_in[GetLinearizedPosition(curr_pos, height, width)], Pixel::RED)]++;
			count_array_green[GetPixelElement(map_in[GetLinearizedPosition(curr_pos, height, width)], Pixel::GREEN)]++;
			count_array_blue[GetPixelElement(map_in[GetLinearizedPosition(curr_pos, height, width)], Pixel::BLUE)]++;
			/*count_array_green[map_in[GetLinearizedPosition(curr_pos, height, width)].Green]++;
			count_array_blue[map_in[GetLinearizedPosition(curr_pos, height, width)].Blue]++;*/

			size++;
		}
	}

	//cuPrintf(" %d-%d-%d-%d-%d\n", count_array_red[0], count_array_red[1], count_array_red[2], count_array_red[3], count_array_red[4]);

	
	uint32_t res = MakePixel(GetMedianElementFromCountArray(count_array_red, size),
		GetMedianElementFromCountArray(count_array_green, size),
		GetMedianElementFromCountArray(count_array_blue, size), 0);
	//cuPrintf("\n%d-%d-%d-%d\n", (int32_t) res.Red, (int32_t) res.Green, (int32_t) res.Blue, (int32_t) res.Alpha);
	return res;
}

__device__ void GetNewPixel(Position pos, uint32_t radius, uint32_t height, uint32_t width,
		uint32_t *map_in, uint32_t *map_out) {
	Position start, end;
	start.X = max(pos.X - (int32_t) radius, 0);
	start.Y = max(pos.Y - (int32_t) radius, 0);
	end.X = min(pos.X + (int32_t) radius, width - 1);
	end.Y = min(pos.Y + (int32_t) radius, height - 1);


	int32_t pos_linear = GetLinearizedPosition(pos, height, width);
	map_out[pos_linear] = GetMedianValue(map_in, height, width, start, end);
}

/*
======
GLOBAL
======
*/

__global__ void MedianFilter(uint32_t radius, uint32_t height, uint32_t width,
		uint32_t *map_in, uint32_t *map_out) {

	Position start, offset;
	start.X = blockIdx.x * blockDim.x + threadIdx.x;
	start.Y = blockIdx.y * blockDim.y + threadIdx.y;

	offset.X = gridDim.x * blockDim.x;
	offset.Y = gridDim.y * blockDim.y;

	//cuPrintf("\n%d:%d\n%d:%d\n======\n", pos.X, pos.Y, height, width);
	Position pos;
	for (pos.X = start.X; pos.X < width; pos.X += offset.X) {
		for (pos.Y = start.Y; pos.Y < height; pos.Y += offset.Y) {
			if (pos.X < width && pos.Y < height) {
				//cuPrintf("\n%d:%d\n", pos.X, pos.Y);
				GetNewPixel(pos, radius, height, width, map_in, map_out);
			}
		}
	}
	//cuPrintf("Hello world from CUDA\n");
}

/*
====
HOST
====
*/

__host__ uint32_t SetPixel(uint8_t red, uint8_t green, uint8_t blue, uint8_t alpha) {
	return ((uint32_t) red << Pixel::RED) + ((uint32_t) green << Pixel::GREEN) +
			((uint32_t) blue << Pixel::BLUE) + ((uint32_t) alpha << Pixel::ALPHA);
}

__host__ void InitPixelMap(uint32_t **pixel, uint32_t height, uint32_t width) {
	*pixel = new uint32_t[height * width];
}

__host__ void DestroyPixelMap(uint32_t **pixel) {
	delete [] (*pixel);
	*pixel = NULL;
}	

__host__ void ReadImageFromFile(uint32_t **pixel, uint32_t *height, uint32_t *width, string filename) {
	FILE *file = fopen(filename.c_str(), "rb");
	fread(width, sizeof(uint32_t), 1, file);
	fread(height, sizeof(uint32_t), 1, file);

	uint32_t size = (*height) * (*width);

	InitPixelMap(pixel, *height, *width);
	fread(*pixel, sizeof(uint32_t), size, file);
	fclose(file);
}

__host__ void WriteImageToFile(uint32_t *pixel, uint32_t height, uint32_t width, string filename) {
	FILE *file = fopen(filename.c_str(), "wb");
	fwrite(&width, sizeof(uint32_t), 1, file);
	fwrite(&height, sizeof(uint32_t), 1, file);

	uint32_t size = height * width;
	fwrite(pixel, sizeof(uint32_t), size, file);
	fclose(file);
}

__host__ void FileGenerator() {
	uint32_t *pixel;
	uint32_t height = 3;
	uint32_t width = 3;
	InitPixelMap(&pixel, height, width);

	string filename = "in.data";
	pixel[0] = SetPixel(1, 2, 3, 0);
	pixel[1] = SetPixel(4, 5, 6, 0);
	pixel[2] = SetPixel(7, 8, 9, 0);

	pixel[3] = SetPixel(9, 8, 7, 0);
	pixel[4] = SetPixel(6, 5, 4, 0);
	pixel[5] = SetPixel(3, 2, 1, 0);

	pixel[6] = SetPixel(0, 0, 0, 0);
	pixel[7] = SetPixel(20, 20, 20, 0);
	pixel[8] = SetPixel(0, 0, 0, 0);

	WriteImageToFile(pixel, height, width, filename);
	DestroyPixelMap(&pixel);
}
__host__ void FileGeneratorBig(uint32_t height, uint32_t width, string filename) {
	uint32_t *pixel;
	InitPixelMap(&pixel, height, width);

	for (uint32_t i = 0; i < height; i++) {
		for (uint32_t j = 0; j < width; j++) {
			uint8_t curr;
			if (i == 0 || j == 0 || i == height - 1 || j == width - 1) {
				curr = 1;
			} else {
				curr = 3;
			}
			pixel[i * width + j] = SetPixel(curr, curr, curr, 0);
		}
	}

	WriteImageToFile(pixel, height, width, filename);
	DestroyPixelMap(&pixel);
}

__host__ int main(void) {
	//FileGeneratorBig(5, 3, "inbig.data");
	string file_in, file_out;
	uint32_t radius;

	cin >> file_in >> file_out >> radius;
	//FileGenerator();
	uint32_t *pixel_in;
	uint32_t *pixel_out;
	uint32_t height, width;
	ReadImageFromFile(&pixel_in, &height, &width, file_in);

	InitPixelMap(&pixel_out, height, width);

	uint32_t *cuda_pixel_in;
	uint32_t *cuda_pixel_out;

	/*size_t pitch;
	cudaMallocPitch((void**) &cuda_pixel_in, &pitch, width * sizeof(Pixel), height);
	cudaMallocPitch((void**) &cuda_pixel_out, &pitch, width * sizeof(Pixel), height);*/
	cudaMalloc((void**) &cuda_pixel_in, sizeof(uint32_t) * width * height);
	cudaMalloc((void**) &cuda_pixel_out, sizeof(uint32_t) * width * height);
	cudaMemcpy(cuda_pixel_in, pixel_in, sizeof(uint32_t) * width * height, cudaMemcpyHostToDevice);


	dim3 threads_per_block(width, height);
	dim3 blocks_per_grid(1, 1);

	if (height * width > BLOCK_DIM * BLOCK_DIM){
		threads_per_block.x = BLOCK_DIM;
		threads_per_block.y = BLOCK_DIM;
		blocks_per_grid.x = ceil((double) (width) / (double)(threads_per_block.x));
		blocks_per_grid.y = ceil((double) (height) / (double)(threads_per_block.y));
	}

	//cout << threads_per_block.x << " " << threads_per_block.y << endl;
	//cout << blocks_per_grid.x << " " << blocks_per_grid.y << endl;

	//cudaPrintfInit();
	MedianFilter<<<blocks_per_grid, threads_per_block>>>(radius, height, width, cuda_pixel_in, cuda_pixel_out);

	//cout << cudaGetErrorString(cudaGetLastError()) << endl;


	//cudaPrintfDisplay(stdout, true);
    //cudaPrintfEnd();



	cudaEvent_t syncEvent;

	cudaEventCreate(&syncEvent);
	cudaEventRecord(syncEvent, 0);
	cudaEventSynchronize(syncEvent);

	cudaMemcpy(pixel_out, cuda_pixel_out, sizeof(uint32_t) * width * height, cudaMemcpyDeviceToHost);

	cudaEventDestroy(syncEvent);

	cudaFree(cuda_pixel_in);
	cudaFree(cuda_pixel_out);

	WriteImageToFile(pixel_out, height, width, file_out.c_str());

	DestroyPixelMap(&pixel_in);
	DestroyPixelMap(&pixel_out);

	return 0;
}
