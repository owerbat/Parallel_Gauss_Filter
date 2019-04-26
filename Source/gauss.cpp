#define _USE_MATH_DEFINES


#include "gauss.h"
#include <iostream>
#include <cmath>


void set_color(const cv::Mat& src, cv::Mat& res, int radius, const Matrix& kernel, int x, int y) {
	int r = 0, g = 0, b = 0;

	for (int i = -radius; i <= radius; ++i)
		for (int j = -radius; j <= radius; ++j) {
			cv::Vec3b neighborColor = src.at<cv::Vec3b>(x + i, y + j);

			r += neighborColor[2] * kernel[j + radius][i + radius];
			g += neighborColor[1] * kernel[j + radius][i + radius];
		    b += neighborColor[0] * kernel[j + radius][i + radius];
		}

	res.at<cv::Vec3b>(x, y) = cv::Vec3b((uchar)b, (uchar)g, (uchar)r);
}


Matrix Gauss::calculate_kernel(float sigma) {
	int size = 2*radius + 1;
	float norm_coef = 0;

	Matrix result{size, Vector(static_cast<size_t>(size))};

    for (int i = -radius; i <= radius; ++i)
        for (int j = -radius; j <= radius; ++j) {
			float current = std::exp(-(i*i + j*j) / (2*sigma*sigma)) / (2*M_PI*sigma*sigma);
			result[i+radius][j+radius] = current;
			norm_coef += current;
		}

	for (int i = 0; i < size; ++i)
		for (int j = 0; j < size; ++j)
			result[i][j] /= norm_coef;	

	return result;
}


void SimpleGauss::gauss_filter() {
	for (int x = radius; x < src.rows - radius; ++x)
		for (int y = radius; y < src.cols - radius; ++y)
			set_color(src, res, radius, kernel, x, y);
}


void OMPGauss::gauss_filter() {
	#pragma omp parallel for schedule(static)
	for (int y = radius; y < src.cols - radius; y++)
		for (int x = radius; x < src.rows - radius; ++x) 
			set_color(src, res, radius, kernel, x, y);
}


void TBBGauss::gauss_filter() {
	tbb::task_scheduler_init init;

    tbb::parallel_for(tbb::blocked_range<int>(radius, src.cols-radius), [this](auto& range) {
		for (auto y = range.begin(); y != range.end(); ++y)
			for (int x = radius; x < src.rows - radius; ++x) 
				set_color(src, res, radius, kernel, x, y);
    });
}


void TBBGaussWithTask::gauss_filter() {
	tbb::task_scheduler_init init;

	BaseTask& root = *new(tbb::task::allocate_root()) BaseTask(src, res, radius, kernel);
	tbb::task::spawn_root_and_wait(root);
}


std::vector<int> BaseTask::get_borders() const {
	int threads_num = tbb::task_scheduler_init::default_num_threads();
	int cols_num = src.cols;

	int average = cols_num / threads_num;
	int tail = cols_num % threads_num;

	std::vector<int> borders;
	borders.emplace_back(0);
	borders.emplace_back(average + tail);
	for (int i = 1; i < threads_num; ++i)
		borders.emplace_back(borders[i] + average);
	
	return borders;
}


tbb::task* BaseTask::execute() {
	tbb::task_list tasks;
	int threads_num = tbb::task_scheduler_init::default_num_threads();
	std::vector<int> borders = get_borders();

	for (int i = 0; i < threads_num; ++i) {
		GaussTask& task = *new(tbb::task::allocate_child()) GaussTask(src, res, std::max(0, borders[i] - 2*radius), borders[i+1], radius, kernel);
		tasks.push_back(task);
	}

	tbb::task::set_ref_count(threads_num + 1);
	tbb::task::spawn_and_wait_for_all(tasks);

	return nullptr;
}


tbb::task* GaussTask::execute() {
	for (int y = min_column + radius; y < max_column - radius; ++y)
		for (int x = radius; x < src.rows - radius; ++x)
			set_color(src, res, radius, kernel, x, y);

	return nullptr;
}
