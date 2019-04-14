#include "gauss.h"
#include <iostream>


void set_color(const cv::Mat& src, cv::Mat& res, const Matrix& kernel, int x, int y) {
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
			set_color(src, res, kernel, x, y);
}


void OMPGauss::gauss_filter() {
	#pragma omp parallel for schedule(static)
	for (int y = 1; y < src.cols - 1; y++)
			for (int x = 1; x < src.rows - 1; ++x) 
				set_color(src, res, kernel, x, y);
}


void TBBGauss::gauss_filter() {
	tbb::task_scheduler_init init;

    tbb::parallel_for(tbb::blocked_range<int>(1, src.cols-1), [this](auto &range) {
		for (auto y = range.begin(); y != range.end(); ++y)
			for (int x = 1; x < src.rows - 1; ++x) 
				set_color(src, res, kernel, x, y);
    });
}


void TBBGaussWithTask::gauss_filter() {
	tbb::task_scheduler_init init;

	BaseTask& root = *new(tbb::task::allocate_root()) BaseTask(src, res);
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
		GaussTask& task = *new(tbb::task::allocate_child()) GaussTask(src, res, std::max(0, borders[i] - 2), borders[i+1]);
		tasks.push_back(task);
	}

	tbb::task::set_ref_count(threads_num + 1);
	tbb::task::spawn_and_wait_for_all(tasks);

	return nullptr;
}


tbb::task* GaussTask::execute() {
	for (int y = min_column + 1; y < max_column - 1; ++y)
		for (int x = 1; x < src.rows - 1; ++x)
			set_color(src, res, kernel, x, y);

	return nullptr;
}
