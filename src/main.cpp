#include "gauss.h"
#include <iostream>
#include <fstream>


bool is_file_exist(const std::string &filename);
std::string get_short_name(const std::string &filename);


int main(int argc, char *argv[]) {
    std::string img_path;

	if (argc > 1) img_path = (std::string)argv[1];
	else img_path = "../Images/img1.png";

    std::string short_name = get_short_name(img_path);
    
    SimpleGauss simple_gauss(img_path);

    double begin_time = clock() / (double)CLOCKS_PER_SEC;
    simple_gauss.gauss_filter();
    printf("Simple time: %lf sec\n", clock() / (double)CLOCKS_PER_SEC - begin_time);

    if (!is_file_exist("../Results/img/simple_" + short_name))
        cv::imwrite("../Results/img/simple_" + short_name, simple_gauss.get_result());

    OMPGauss omp_gauss(img_path);

    double omp_begin_time = omp_get_wtime();
    omp_gauss.gauss_filter();
    printf("OpenMP time: %lf sec\n", omp_get_wtime() - omp_begin_time);

    if (!is_file_exist("../Results/img/omp_" + short_name))
        cv::imwrite("../Results/img/omp_" + short_name, omp_gauss.get_result());

    TBBGauss tbb_gauss(img_path);

    tbb::tick_count tbb_begin_time = tbb::tick_count::now();
    tbb_gauss.gauss_filter();
    printf("TBB time: %g sec\n", (tbb::tick_count::now() - tbb_begin_time).seconds());

    if (!is_file_exist("../Results/img/tbb_" + short_name))
        cv::imwrite("../Results/img/tbb_" + short_name, tbb_gauss.get_result());

    return 0;
}


bool is_file_exist(const std::string &filename) {
    std::ifstream infile(filename.c_str());
    return infile.good();
}

std::string get_short_name(const std::string &filename) {
    std::string slash = filename.rfind("\\") != std::string::npos ? "\\" : "/";
	size_t position = filename.rfind(slash) + 1;
	std::string result = "";

	for (unsigned i = position; i < filename.length(); ++i)
		result += filename[i];

	return result;
}
