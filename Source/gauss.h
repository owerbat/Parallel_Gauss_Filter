#include <opencv4/opencv2/opencv.hpp>
#include <time.h>

#include <omp.h>

#include <tbb/task_scheduler_init.h>
#include <tbb/parallel_for.h>
#include <tbb/tick_count.h>
#include <tbb/task.h>


using Vector = std::vector<float>;
using Matrix = std::vector<Vector>;


void set_color(const cv::Mat& src, cv::Mat& res, int radius, const Matrix& kernel, int x, int y);


class Gauss {
protected:
    cv::Mat src;
    cv::Mat res;
    int radius;
    Matrix kernel;

    Matrix calculate_kernel(float sigma);

public:
    Gauss(std::string img_path = "../Images/img1.png", int _radius = 2): radius(_radius) {
        src = cv::imread(img_path);
        res = src.clone();
        kernel = calculate_kernel(2.0f);
    }
    cv::Mat& get_result() { return res; }
    virtual void gauss_filter() = 0;
};


class SimpleGauss: public Gauss {
public:
    SimpleGauss(std::string img_path = "../Images/img1.png", int _radius = 2): Gauss(img_path, _radius) {}
    void gauss_filter();
};


class OMPGauss: public Gauss {
public:
    OMPGauss(std::string img_path = "../Images/img1.png", int _radius = 2): Gauss(img_path, _radius) {}
    void gauss_filter();
};


class TBBGauss: public Gauss {
public:
    TBBGauss(std::string img_path = "../Images/img1.png", int _radius = 2): Gauss(img_path, _radius) {}
    void gauss_filter();
};


class TBBGaussWithTask: public Gauss {
public:
    TBBGaussWithTask(std::string img_path = "../Images/img1.png", int _radius = 2): Gauss(img_path, _radius) {}
    void gauss_filter();
};


class BaseTask: public tbb::task {
	cv::Mat src;
	cv::Mat res;
    int radius;
    Matrix kernel;

    std::vector<int> get_borders() const;

public:
	BaseTask(const cv::Mat& _src, cv::Mat& _res, int _radius, Matrix _kernel):
        src(_src), res(_res), radius(_radius), kernel(_kernel) {}
    tbb::task* execute();
};


class GaussTask: public tbb::task {
private:
	cv::Mat src;
	cv::Mat res;
	int min_column, max_column;
    int radius;
	Matrix kernel;

public:
    GaussTask(const cv::Mat& _src, cv::Mat& _res, int min, int max, int _radius, Matrix _kernel):
        src(_src), res(_res), min_column(min), max_column(max), radius(_radius), kernel(_kernel) {}
    tbb::task* execute();
};
