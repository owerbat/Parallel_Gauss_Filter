#include <opencv4/opencv2/opencv.hpp>
#include <time.h>

#include <omp.h>

#include <tbb/task_scheduler_init.h>
#include <tbb/parallel_for.h>
#include <tbb/tick_count.h>


using Matrix = std::vector<std::vector<float>>;


class Gauss {
protected:
    cv::Mat src;
    cv::Mat res;
    Matrix kernel;
    void set_color(int x, int y);
public:
    Gauss(std::string img_path = "../Images/img1.png") {
        src = cv::imread(img_path);
        res = src.clone();
        const Matrix kernel = { std::vector<float>({0.0625f, 0.125f, 0.0625f}),
					  		    std::vector<float>({0.125f, 0.25f, 0.125f}),
					  		    std::vector<float>({0.0625f, 0.125f, 0.0625f}) };
    }
    virtual void gauss_filter() = 0;
};


class SimpleGauss: public Gauss {
public:
    SimpleGauss(std::string img_path = "../Images/img1.png"): Gauss(img_path) {}
    void gauss_filter();
};


class OMPGauss: public Gauss {
public:
    OMPGauss(std::string img_path = "../Images/img1.png"): Gauss(img_path) {}
    void gauss_filter();
};


class TBBGauss: public Gauss {
public:
    TBBGauss(std::string img_path = "../Images/img1.png"): Gauss(img_path) {}
    void gauss_filter();
};
