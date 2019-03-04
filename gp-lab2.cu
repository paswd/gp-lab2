#include <iostream>
#include <algorithm>
#include <string>
#include <cstdio>
#include <cstdlib>

using namespace std;

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

__device__ double GetIntensity(Pixel pixel) {
	return (.3 * (double) pixel.Red) + (.59 * (double) pixel.Green) + (.11 * (double) pixel.Blue);
}

__device__ bool IsCorrectPos(Position pos, uint32_t height, uint32_t width) {
	if (pos.X >= 0 && pos.Y >= 0 && pos.X < (int32_t) height && pos.Y < (int32_t) width) {
		return true;
	}
	return false;
}

__device__ void GetNewPixel(Position pos, uint32_t radius, uint32_t height, uint32_t width,
		Pixel **map_in, Pixel **map_out) {
	Position start, end;
	start.X = pos.X - (int32_t) radius;
	start.Y = pos.Y - (int32_t) radius;
	end.X = pos.X + (int32_t) radius;
	end.Y = pos.Y + (int32_t) radius;

	uint32_t kernel_width = (radius * 2 + 1);
	uint32_t kernel_array_size = kernel_width * kernel_width;

	uint8_t *kernel_array_red = new uint8_t[kernel_array_size];
	uint8_t *kernel_array_green = new uint8_t[kernel_array_size];
	uint8_t *kernel_array_blue = new uint8_t[kernel_array_size];

	uint32_t curr = 0;
	Position curr_pos;

	for (curr_pos.X = start.X; curr_pos.X < end.X; curr_pos.X++) {
		for (curr_pos.Y = start.Y; curr_pos.Y < end.Y; curr_pos.Y++) {
			if (!IsCorrectPos(curr_pos, height, width)) {
				continue;
			}
			//kernel_array[curr] = map_in[curr_pos.X][curr_pos.Y];
			kernel_array_red[curr] = map_in[curr_pos.X][curr_pos.Y].Red;
			kernel_array_green[curr] = map_in[curr_pos.X][curr_pos.Y].Green;
			kernel_array_blue[curr] = map_in[curr_pos.X][curr_pos.Y].Blue;
			curr++;
		}
	}
	sort(kernel_array_red, kernel_array_red + curr);
	sort(kernel_array_green, kernel_array_green + curr);
	sort(kernel_array_blue, kernel_array_blue + curr);

	map_out[pos.X][pos.Y].Red = kernel_array_red[curr / 2];
	map_out[pos.X][pos.Y].Green = kernel_array_green[curr / 2];
	map_out[pos.X][pos.Y].Blue = kernel_array_blue[curr / 2];

	delete [] kernel_array_red;
	delete [] kernel_array_green;
	delete [] kernel_array_blue;
}

__global__ void MedianFilter(uint32_t radius, uint32_t height, uint32_t width,
		Pixel **map_in, Pixel **map_out) {
	Position begin, offset;
	begin.X = (int32_t) (blockDim.x * blockIdx.x + threadIdx.x);
	offset.X = (int32_t) (gridDim.x * blockDim.x);
	begin.Y = (int32_t) (blockDim.y * blockIdx.y + threadIdx.y);
	offset.Y = (int32_t) (gridDim.y * blockDim.y);

	Position pos;

	for (pos.X = begin.X; pos.X < height; pos.X += offset.X) {
		for (pos.Y = begin.Y; pos.Y < width; pos.Y += offset.Y) {
			GetNewPixel(pos, radius, height, width, map_in, map_out);
		}
	}
}

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

__host__ void InitPixelMap(Pixel ***pixel, uint32_t height, uint32_t width) {
	*pixel = new Pixel*[height];
	for (uint32_t i = 0; i < height; i++) {
		(*pixel)[i] = new Pixel[width];
	}
}

__host__ void DestroyPixelMap(Pixel ***pixel, uint32_t height) {
	for (uint32_t i = 0; i < height; i++) {
		delete [] (*pixel)[i];
	}
	delete [] (*pixel);
	*pixel = NULL;
}	

__host__ void ReadImageFromFile(Pixel ***pixel, uint32_t *height, uint32_t *width, string filename) {
	FILE *file = fopen(filename.c_str(), "rb");
	fread(width, sizeof(uint32_t), 1, file);
	fread(height, sizeof(uint32_t), 1, file);

	InitPixelMap(pixel, *height, *width);
	for (uint32_t i = 0; i < *height; i++) {
		for (uint32_t j = 0; j < *width; j++) {
			ReadPixelFromFile(&((*pixel)[i][j]), file);
		}
	}
	fclose(file);
}

__host__ void WriteImageToFile(Pixel **pixel, uint32_t height, uint32_t width, string filename) {
	FILE *file = fopen(filename.c_str(), "wb");
	fwrite(&width, sizeof(uint32_t), 1, file);
	fwrite(&height, sizeof(uint32_t), 1, file);

	for (uint32_t i = 0; i < height; i++) {
		for (uint32_t j = 0; j < width; j++) {
			WritePixelToFile(&(pixel)[i][j], file);
		}
	}
}

__host__ void FileGenerator() {
	Pixel **pixel;
	uint32_t height = 3;
	uint32_t width = 3;
	InitPixelMap(&pixel, height, width);

	string filename = "in1.data";
	pixel[0][0] = SetPixel(1, 2, 3, 0);
	pixel[0][1] = SetPixel(4, 5, 6, 0);
	pixel[0][2] = SetPixel(7, 8, 9, 0);

	pixel[1][0] = SetPixel(9, 8, 7, 0);
	pixel[1][1] = SetPixel(6, 5, 4, 0);
	pixel[1][2] = SetPixel(3, 2, 1, 0);

	pixel[2][0] = SetPixel(0, 0, 0, 0);
	pixel[2][1] = SetPixel(20, 20, 20, 0);
	pixel[2][2] = SetPixel(0, 0, 0, 0);

	WriteImageToFile(pixel, height, width, filename);
	DestroyPixelMap(&pixel, height);
}

__host__ int main(void) {
	/*string file_in, file_out;
	uint32_t radius;

	cin >> file_in >> file_out >> radius;*/
	FileGenerator();

	return 0;
}

/*__host__ int mainOld(void) {
	size_t size;
	cin >> size;

	double *first = new double[size];
	double *second = new double[size];
	double *res = new double[size];

	for (size_t i = 0; i < size; i++) {
		cin >> first[i];
		//first[i] = i;
	}
	for (size_t i = 0; i < size; i++) {
		cin >> second[i];
		//second[i] = i;
	}

	double *cudaFirst;
	double *cudaSecond;
	double *cudaRes;

	cudaMalloc((void**) &cudaFirst, sizeof(double) * size);
	cudaMalloc((void**) &cudaSecond, sizeof(double) * size);
	cudaMalloc((void**) &cudaRes, sizeof(double) * size);

	cudaMemcpy(cudaFirst, first, sizeof(double) * size, cudaMemcpyHostToDevice);
	cudaMemcpy(cudaSecond, second, sizeof(double) * size, cudaMemcpyHostToDevice);

	VectorsPairMaximums<<<256, 256>>>(size, cudaFirst, cudaSecond, cudaRes);

	cudaEvent_t syncEvent;

	cudaEventCreate(&syncEvent);
	cudaEventRecord(syncEvent, 0);
	cudaEventSynchronize(syncEvent);

	cudaMemcpy(res, cudaRes, sizeof(double) * size, cudaMemcpyDeviceToHost);

	//double *testArr = new double[size];
	//cudaMemcpy(testArr, cudaFirst, sizeof(double) * size, cudaMemcpyDeviceToHost);

	cudaEventDestroy(syncEvent);
	cudaFree(cudaFirst);
	cudaFree(cudaSecond);
	cudaFree(cudaRes);

	for (size_t i = 0; i < size; i++) {
		if (i > 0) {
			cout << " ";
		}
		cout << scientific << res[i];
	}
	cout << endl;

	delete [] first;
	delete [] second;
	delete [] res;

	return 0;
}*/