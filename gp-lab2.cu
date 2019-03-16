#include <iostream>
#include <algorithm>
#include <string>
#include <cstdio>
#include <cstdlib>
#include "../lib/cuPrintf.cu"

using namespace std;

/*
=========
CONSTANTS
=========
*/

const uint32_t COUNTING_SORT_BASE = 256;
const uint32_t BLOCK_DIM = 32;

/*
==========
STRUCTURES
==========
*/

struct Pixel {
	uint8_t Red;
	uint8_t Green;
	uint8_t Blue;
	uint8_t Alpha;
};

struct Position {
	int32_t X;
	int32_t Y;
};

/*
======
DEVICE
======
*/

__device__ double GetIntensity(Pixel pixel) {
	return (.3 * (double) pixel.Red) + (.59 * (double) pixel.Green) + (.11 * (double) pixel.Blue);
}

__device__ bool IsCorrectPos(Position pos, uint32_t height, uint32_t width) {
	if (pos.X >= 0 && pos.Y >= 0 && pos.X < (int32_t) height && pos.Y < (int32_t) width) {
		return true;
	}
	return false;
}

/*__device__ void CountingSort(uint8_t *array, uint32_t size) {
	uint32_t count_array[COUNTING_SORT_BASE];
	for (uint32_t i = 0; i < COUNTING_SORT_BASE; i++) {
		count_array[i] = 0;
	}
	for (uint32_t i = 0; i < size; i++) {
		count_array[array[i]]++;
	}
	uint32_t current = 0;
	for (uint32_t i = 0; i < COUNTING_SORT_BASE; i++) {
		for (uint32_t j = 0; j < count_array[i]; j++) {
			array[current] = i;
			current++;
		}
	}
}*/

__device__ int32_t GetLinearizedPosition(Position pos, uint32_t height, uint32_t width) {
	if (!IsCorrectPos(pos, height, width)) {
		return -1;
	}
	return pos.Y * width + pos.X;
}

__device__ Pixel GetMedianValue(Pixel *map_in, uint32_t height, uint32_t width, Position start, Position end) {
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

	for (curr_pos.X = start.X; curr_pos.X <= end.X; curr_pos.X++) {
		for (curr_pos.Y = start.Y; curr_pos.Y <= end.Y; curr_pos.Y++) {
			if (!IsCorrectPos(curr_pos, height, width)) {
				continue;
			}
			count_array_red[map_in[GetLinearizedPosition(curr_pos, height, width)].Red]++;
			count_array_green[map_in[GetLinearizedPosition(curr_pos, height, width)].Green]++;
			count_array_blue[map_in[GetLinearizedPosition(curr_pos, height, width)].Blue]++;

			size++;
		}
	}

	Pixel res;
	uint32_t tmp_cnt = 0;
	bool is_break = false;
	for (uint32_t i = 0; i < COUNTING_SORT_BASE; i++) {
		for (uint32_t j = 0; j < count_array_red[i]; j++) {
			if (tmp_cnt == size / 2) {
				res.Red = i;
				is_break = true;
				break;
			}
			tmp_cnt++;
		}
		if (is_break) {
			break;
		}
	}
	tmp_cnt = 0;
	is_break = false;
	for (uint32_t i = 0; i < COUNTING_SORT_BASE; i++) {
		for (uint32_t j = 0; j < count_array_green[i]; j++) {
			if (tmp_cnt == size / 2) {
				res.Green = i;
				is_break = true;
				break;
			}
			tmp_cnt++;
		}
		if (is_break) {
			break;
		}
	}
	tmp_cnt = 0;
	is_break = false;
	for (uint32_t i = 0; i < COUNTING_SORT_BASE; i++) {
		for (uint32_t j = 0; j < count_array_blue[i]; j++) {
			if (tmp_cnt == size / 2) {
				res.Blue = i;
				is_break = true;
				break;
			}
			tmp_cnt++;
		}
		if (is_break) {
			break;
		}
	}
	res.Alpha = 0;
	return res;
}

__device__ void GetNewPixel(Position pos, uint32_t radius, uint32_t height, uint32_t width,
		Pixel *map_in, Pixel *map_out) {
	Position start, end;
	start.X = pos.X - (int32_t) radius;
	start.Y = pos.Y - (int32_t) radius;
	end.X = pos.X + (int32_t) radius;
	end.Y = pos.Y + (int32_t) radius;


	int32_t pos_linear = GetLinearizedPosition(pos, height, width);
	/*map_out[pos_linear].Red = map_in[pos_linear].Red;
	map_out[pos_linear].Green = map_in[pos_linear].Green;
	map_out[pos_linear].Blue = map_in[pos_linear].Blue;*/
	map_out[pos_linear] = GetMedianValue(map_in, height, width, start, end);
}

/*
======
GLOBAL
======
*/

__global__ void MedianFilter(uint32_t radius, uint32_t height, uint32_t width,
		Pixel *map_in, Pixel *map_out) {
	/*Position begin, offset;
	begin.X = (int32_t) (blockDim.x * blockIdx.x + threadIdx.x);
	offset.X = (int32_t) (gridDim.x * blockDim.x);
	begin.Y = (int32_t) (blockDim.y * blockIdx.y + threadIdx.y);
	offset.Y = (int32_t) (gridDim.y * blockDim.y);*/

	Position pos;
	pos.X = blockIdx.x * blockDim.x + threadIdx.x;
	pos.Y = blockIdx.y * blockDim.y + threadIdx.y;
	//cuPrintf("TEST2\n");

	/*for (pos.X = begin.X; pos.X < height; pos.X += offset.X) {
		for (pos.Y = begin.Y; pos.Y < width; pos.Y += offset.Y) {
			GetNewPixel(pos, radius, height, width, map_in, map_out);
		}
	}*/

	//cuPrintf("\n%d:%d\n%d:%d\n======\n", pos.X, pos.Y, height, width);
	if (pos.X < width && pos.Y < height) {
		GetNewPixel(pos, radius, height, width, map_in, map_out);
	}
	//cuPrintf("Hello world from CUDA\n");
}

/*
====
HOST
====
*/

__host__ Pixel SetPixel(uint8_t red, uint8_t green, uint8_t blue, uint8_t alpha) {
	Pixel pixel;
	pixel.Red = red;
	pixel.Green = green;
	pixel.Blue = blue;
	pixel.Alpha = alpha;

	return pixel;
}

__host__ void ReadPixelFromFile(Pixel *pixel, FILE *file) {
	fread(&(pixel->Red), sizeof(uint8_t), 1, file);
	fread(&(pixel->Green), sizeof(uint8_t), 1, file);
	fread(&(pixel->Blue), sizeof(uint8_t), 1, file);
	fread(&(pixel->Alpha), sizeof(uint8_t), 1, file);
}

__host__ void WritePixelToFile(Pixel *pixel, FILE *file) {
	fwrite(&(pixel->Red), sizeof(uint8_t), 1, file);
	fwrite(&(pixel->Green), sizeof(uint8_t), 1, file);
	fwrite(&(pixel->Blue), sizeof(uint8_t), 1, file);
	fwrite(&(pixel->Alpha), sizeof(uint8_t), 1, file);
}

__host__ void InitPixelMap(Pixel **pixel, uint32_t height, uint32_t width) {
	*pixel = new Pixel[height * width];
	/*for (uint32_t i = 0; i < height; i++) {
		(*pixel)[i] = new Pixel[width];
	}*/
}

__host__ void DestroyPixelMap(Pixel **pixel) {
	/*for (uint32_t i = 0; i < height; i++) {
		delete [] (*pixel)[i];
	}*/
	delete [] (*pixel);
	*pixel = NULL;
}	

__host__ void ReadImageFromFile(Pixel **pixel, uint32_t *height, uint32_t *width, string filename) {
	FILE *file = fopen(filename.c_str(), "rb");
	fread(width, sizeof(uint32_t), 1, file);
	fread(height, sizeof(uint32_t), 1, file);

	InitPixelMap(pixel, *height, *width);
	/*for (uint32_t i = 0; i < *height; i++) {
		for (uint32_t j = 0; j < *width; j++) {
			ReadPixelFromFile(&((*pixel)[i][j]), file);
		}
	}*/
	for (uint32_t i = 0; i < (*height) * (*width); i++) {
		ReadPixelFromFile(&((*pixel)[i]), file);
	}
	fclose(file);
}

__host__ void WriteImageToFile(Pixel *pixel, uint32_t height, uint32_t width, string filename) {
	FILE *file = fopen(filename.c_str(), "wb");
	fwrite(&width, sizeof(uint32_t), 1, file);
	fwrite(&height, sizeof(uint32_t), 1, file);

	/*for (uint32_t i = 0; i < height; i++) {
		for (uint32_t j = 0; j < width; j++) {
			WritePixelToFile(&(pixel)[i][j], file);
		}
	}*/
	for (uint32_t i = 0; i < height * width; i++) {
		WritePixelToFile(&(pixel)[i], file);
	}
	fclose(file);
}

__host__ void FileGenerator() {
	Pixel *pixel;
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
	Pixel *pixel;
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
	FileGeneratorBig(128, 128, "inbig.data");
	string file_in, file_out;
	uint32_t radius;

	cin >> file_in >> file_out >> radius;
	//FileGenerator();
	Pixel *pixel_in;
	Pixel *pixel_out;
	uint32_t height, width;
	ReadImageFromFile(&pixel_in, &height, &width, file_in);
	//WriteImageToFile(pixel, height, width, "out.data");

	InitPixelMap(&pixel_out, height, width);

	Pixel *cuda_pixel_in;
	Pixel *cuda_pixel_out;

	/*size_t pitch;
	cudaMallocPitch((void**) &cuda_pixel_in, &pitch, width * sizeof(Pixel), height);
	cudaMallocPitch((void**) &cuda_pixel_out, &pitch, width * sizeof(Pixel), height);*/
	cudaMalloc((void**) &cuda_pixel_in, sizeof(Pixel) * width * height);
	cudaMalloc((void**) &cuda_pixel_out, sizeof(Pixel) * width * height);
	cudaMemcpy(cuda_pixel_in, pixel_in, sizeof(Pixel) * width * height, cudaMemcpyHostToDevice);

	/*dim3 grid_size = dim3((height / BLOCK_DIM) + 1, (width / BLOCK_DIM) + 1, 1);
	dim3 block_size = dim3(BLOCK_DIM, BLOCK_DIM, 1);*/

	dim3 threads_per_block(width, height);
	dim3 blocks_per_grid(1, 1);

	if (height * width > BLOCK_DIM * BLOCK_DIM){
		threads_per_block.x = BLOCK_DIM;
		threads_per_block.y = BLOCK_DIM;
		blocks_per_grid.x = ceil((double) (width) / (double)(threads_per_block.x));
		blocks_per_grid.y = ceil((double) (height) / (double)(threads_per_block.y));
	}

	cout << threads_per_block.x << " " << threads_per_block.y << endl;
	cout << blocks_per_grid.x << " " << blocks_per_grid.y << endl;

	//cudaPrintfInit();
	MedianFilter<<<blocks_per_grid, threads_per_block>>>(radius, height, width, cuda_pixel_in, cuda_pixel_out);

	cout << cudaGetErrorString(cudaGetLastError()) << endl;


	//cudaPrintfDisplay(stdout, true);
    //cudaPrintfEnd();



	cudaEvent_t syncEvent;

	cudaEventCreate(&syncEvent);
	cudaEventRecord(syncEvent, 0);
	cudaEventSynchronize(syncEvent);

	cudaMemcpy(pixel_out, cuda_pixel_out, sizeof(Pixel) * width * height, cudaMemcpyDeviceToHost);

	cudaEventDestroy(syncEvent);

	cudaFree(cuda_pixel_in);
	cudaFree(cuda_pixel_out);

	WriteImageToFile(pixel_out, height, width, file_out.c_str());

	DestroyPixelMap(&pixel_in);
	DestroyPixelMap(&pixel_out);

	return 0;
}
