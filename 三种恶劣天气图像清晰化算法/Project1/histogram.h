#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <opencv2\opencv.hpp>

#include <iostream>
#include <string>

class Histogram
{
public:
	int histSize[3];        //直方图中箱子的数量
	float hranges[2];       //值范围
	const float * ranges[3];        //值范围的指针
	int channels[3];        //要检查的通道数量


	Histogram();
	cv::Mat getHistogram(const cv::Mat & image);
	std::vector<cv::Mat> getHistogramImage(const cv::Mat & image, int zoom = 1);
	static std::vector<cv::Mat> getImageOfHistogram(const cv::Mat & hist, int zoom);
};

#endif
