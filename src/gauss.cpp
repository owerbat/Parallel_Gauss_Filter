#include "gauss.h"


void Gauss::set_color(int x, int y) {
    int r = 0, g = 0, b = 0;

	for (int i = -1; i <= 1; ++i)
		for (int j = -1; j <= 1; ++j) {
			cv::Vec3b neighborColor = src.at<cv::Vec3b>(x + i, y + j);

			r += neighborColor[2] * kernel[j + 1][i + 1];
			g += neighborColor[1] * kernel[j + 1][i + 1];
			b += neighborColor[0] * kernel[j + 1][i + 1];
		}

	res.at<cv::Vec3b>(x, y) = cv::Vec3b((uchar)b, (uchar)g, (uchar)r);
}


void SimpleGauss::gauss_filter() {
	for (int x = 1; x < src.rows - 1; ++x)
		for (int y = 1; y < src.cols - 1; ++y)
			set_color(x, y);
}


void OMPGauss::gauss_filter() {
	#pragma omp parallel for schedule(static)
	for (int y = 1; y < src.cols - 1; y++)
			for (int x = 1; x < src.rows - 1; ++x) 
				set_color(x, y);
}


void TBBGauss::gauss_filter() {
	tbb::task_scheduler_init init;

    tbb::parallel_for(tbb::blocked_range<int>(1, src.cols-1), [this](auto &range) {
		for (auto y = range.begin(); y != range.end(); ++y)
			for (int x = 1; x < src.rows - 1; ++x) 
				set_color(x, y);
    });
}
