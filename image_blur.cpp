#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "image_blur.h"

using namespace std;
using namespace cv;

int main() {
	Mat Input_Image = imread("C:\\Users\\alex_\\source\\repos\\ppd_P2_Cuda\\ppd_P2_Cuda\\Test_Image.png");

	cout << "Height: " << Input_Image.rows << ", Width: " << Input_Image.rows << ", Channels: " << Input_Image.channels() << endl;


	unsigned char* in_image = Input_Image.data;
	

	image_blur_cuda(Input_Image.data, Input_Image.rows, Input_Image.rows, Input_Image.channels());

	imwrite("Blurred_Image.png", Input_Image);
	system("pause");
	return 0;
}