#include "gauss.h"
#include "file_interactions.h"
#include <iostream>
#include <string>


int main(int argc, char *argv[]) {
    int radius = 2;
	if (argc > 1) radius = std::stoi(argv[1]);

    std::string img_path = "../Images/img1.png";
	if (argc > 2) img_path = (std::string)argv[2];
    std::string short_name = get_short_name(img_path);

    //parallel_for version
    TBBGauss tbb_gauss(img_path, radius);

    tbb::tick_count tbb_begin_time = tbb::tick_count::now();
    tbb_gauss.gauss_filter();
    double tbb_time = (tbb::tick_count::now() - tbb_begin_time).seconds();

    //version with tasks
    TBBGaussWithTask tbb_tasks_gauss(img_path, radius);

    tbb::tick_count tbb_tasks_begin_time = tbb::tick_count::now();
    tbb_tasks_gauss.gauss_filter();
    double tbb_tasks_time = (tbb::tick_count::now() - tbb_tasks_begin_time).seconds();

    //output
    std::cout << "parallel_for time: " << tbb_time << std::endl;
    std::cout << "tasks time: " << tbb_tasks_time << std::endl;

    cv::imshow("original", cv::imread(img_path));
    cv::imshow("parallel_for image", tbb_gauss.get_result());
    cv::imshow("tasks image", tbb_tasks_gauss.get_result());

    cv::waitKey(0);

    if (!is_file_exist("../Results/img/tbb_for_" + short_name))
        cv::imwrite("../Results/img/tbb_for_" + short_name, tbb_gauss.get_result());
    
    if (!is_file_exist("../Results/img/tbb_tasks_" + short_name))
        cv::imwrite("../Results/img/tbb_tasks_" + short_name, tbb_tasks_gauss.get_result());

    return 0;
}
