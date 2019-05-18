#include <opencv2\opencv.hpp>
#include <iostream>
#include <string>

using namespace cv;

class Adjustment
{
public:
	int histSize[3];        //直方图中箱子的数量
	float hranges[2];       //值范围
	const float * ranges[3];        //值范围的指针
	int channels[3];        //要检查的通道数量

	/*去沙城暴中会使用的属性*/
	cv::Mat hist_b, hist_g, hist_r;    // 原始图像的三通道矩阵
	//得到总的像素点数，行数，列数
	int num, r, c;
	cv::Mat finalImg;       //得到的最终图像矩阵
	double x[3];
	double y[3];             //三通道的均值和标准差的估计值
	cv::Mat aimg;           //第一阶段后图像矩阵



	//构造函数
	Adjustment();


			/*合成一中算法*/
	Mat de_all(cv::Mat  image, IplImage *Image1);


			/*去沙城暴总方法*/
	Mat de_sand(cv::Mat  image, IplImage *Image1,int flag);
	//通过直方图，求三通道对应估计值
	Mat getvalue2(const cv::Mat  image , IplImage *Image1);
	//去沙尘暴——求取第一阶段图像矩阵
	void getaImg(IplImage *Image1, std::vector<cv::Mat> rgbChannels);
	//去沙尘暴——求取最后的图像矩阵
	void getfinalImg();
	//去沙尘暴——SVD图像增强
	Mat SVD_de_sand(Mat rec, Mat sec_hist);


			/*去雾方法*/
	Mat defog(Mat image,int flag );//暗通道方法

	Mat defog_2(Mat image);//灰度拉伸方法
	Mat grayStretch(cv::Mat src, double lowcut, double highcut);

	Mat defog_3(Mat image,int flag);//卷积滤波方法
	Mat Adaptive_contrast(Mat img);

	Mat defog_4(Mat image, int flag);


			/*低亮度增强*/
	//低亮度增强(去雾方法）
	Mat ad_light(Mat image) ;
	//低亮度增强（对数Log方法）
	Mat ad_light_Log(Mat image);
	//降低亮度
	Mat drop_light(Mat image);

};
