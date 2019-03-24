#include "gauss.h"


int main(int argc, char *argv[]) {
    printf("AAA");
    std::string img_path;

	if (argc > 1) img_path = (std::string)argv[1];
	else img_path = "../Images/img1.png";
    
    SimpleGauss simple_gauss(img_path);

    double begin_time = clock() / (double)CLOCKS_PER_SEC;
    simple_gauss.gauss_filter();
    printf("Time: %lf sec\n", clock() / (double)CLOCKS_PER_SEC - begin_time);

    OMPGauss omp_gauss(img_path);

    double omp_begin_time = omp_get_wtime();
    omp_gauss.gauss_filter();
    printf("OpenMP time: %lf sec\n", omp_get_wtime() - omp_begin_time);

    TBBGauss tbb_gauss(img_path);

    tbb::tick_count tbb_begin_time = tbb::tick_count::now();
    tbb_gauss.gauss_filter();
    printf("TBB time: %g sec\n", (tbb::tick_count::now() - tbb_begin_time).seconds());

    //cv::imshow("Image", simple_gauss);

    return 0;
}
