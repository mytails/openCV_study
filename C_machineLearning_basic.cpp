#include <iostream>
#include <opencv2/opencv.hpp>

#define NUM 100
#define WIDTH 1000
#define HEIGHT 1000
#define ABS(x) ((x < 0) ? -x : x)
#define MIN(a, b) ((a < b) ? a : b)
#define MAX(a, b) ((a > b) ? a : b)

using namespace std;
using namespace cv;

typedef struct Example
{
	float size;
	float price;	
} Example;

Example *example_list;

float weight = 0.0f, bias = 0.0f;
float weight_min = 0.0f, bias_min = 0.0f;
float error_min = (float) 1e9;

float getLinearHypothesis(const float x){
	return weight * x + bias;
}

float getMeanSquaredError(){
	float error_sum = 0.0f;
	for(int i=0;i<NUM;i++){
		const float error = example_list[i].price - getLinearHypothesis(example_list[i].size);
		error_sum += error*error;
	}
	return error_sum/(float)NUM * 0.5f; // last 0.5f multiplication is a custom
}

int main(int argc, char** argv){
	example_list = (Example*)malloc(sizeof(Example) * NUM);
	
	// randomize
	for (int idx = 0; idx < NUM; idx++){
		srand(time(NULL)*idx);
		example_list[idx].size = rand()%1000;
		example_list[idx].price = rand()%30000 + 10000;
	}

	// check
	for(float i=0;i<100;i++){
		weight = i;
		for(float j=0;j<30000;j++){
			bias = j;
			float err = getMeanSquaredError();
			if(err < error_min){
				error_min = err;
				weight_min = weight;
				bias_min = bias;
			}
		}
	}
	weight = weight_min;
	bias = bias_min;

	// check data
	for(int k=0;k<NUM;k++)
		printf("%d (size, price) = (%.1f, %.1f) \n", k, example_list[k].size, example_list[k].price);
	
	// image setting
	IplImage *window = cvCreateImage(cvSize(WIDTH,HEIGHT), 8, 3);
	cvSet(window, cvScalar(255,255,255));

	float rate_y = (float)HEIGHT/50000;

	// draw Y-axis
	for(int x=0;x<10;x++){
		cvSet2D(window, HEIGHT - 10000 * rate_y, x, cvScalar(0,0,255));
		cvSet2D(window, HEIGHT - 20000 * rate_y, x, cvScalar(0,0,255));
		cvSet2D(window, HEIGHT - 30000 * rate_y, x, cvScalar(0,0,255));
		cvSet2D(window, HEIGHT - 40000 * rate_y, x, cvScalar(0,0,255));
	}
	
	// draw point
	for(int s=0;s<NUM;s++){
		int value_y = example_list[s].price * rate_y;
		int value_x = example_list[s].size;
		for(int y=value_y;y<value_y+10;y++){
			for(int x=value_x;x<value_x+10;x++){
				if(value_y+10 < HEIGHT && value_x+10 < WIDTH)
					cvSet2D(window, HEIGHT - y -1, x, cvScalar(0,0,0));
			}
		}
	}

	// draw line
	printf("\nweight_min: %.0f, bias_min: %.0f, error_min: %.0f\n\n", weight_min, bias_min, error_min);
	for(int l=0;l<WIDTH;l++){
		if(getLinearHypothesis(l) * rate_y -1 > HEIGHT)
			break;
		else
			cvSet2D(window, HEIGHT - getLinearHypothesis(l) * rate_y -1, l, cvScalar(255,0,0));
	}

	// show image
	Mat mat = (Mat) window;
	imshow("C_MachineLearning", mat);
	waitKey(0);
	cvReleaseImage(&window);

	free(example_list);
	return 0;
}