#include <iostream> 
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>
#include<string>
#include "histogram.h"
#include"Adjust.h"

using namespace cv;
using namespace std;
int main() {

	const int num = 670;
	char filename[250];
	char windowname[250];
	Adjustment ad;

	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);  //选择jpeg
	compression_params.push_back(100); //在这个填入你要的图片质量


	for (int i = 1; i <= 2038; i++) {
		sprintf_s(filename,250, "C:\\Users\\Desktop\\图片素材数据库\\雾霾\\雾霾%d.jpg", i);
		// 00001.jpg ，00002.jpg等，放入D:/test/文件夹下  		
		sprintf_s(windowname, 250,"C:\\Users\\Desktop\\图片素材数据库\\雾霾_自己的算法\\%d.jpg", i);  //新图片命名文字
		string file = filename;
		string wind = windowname;

		cout << filename << endl;
		cout << windowname << endl;
		//waitKey(0);

		
		Mat image = imread(file);//导入图片 

		//imshow("image",image);

		Mat dst;

		IplImage *Imagel = cvLoadImage(filename, 1);
		dst = ad.de_all(image, Imagel);
		//flip(image,dst,1);	
		//namedWindow(windowname,1);  
		dst.convertTo(dst, CV_8UC3,255,0);
		//imshow(wind,dst);//显示图片 		
		imwrite(wind, dst, compression_params);
		cout << wind << "已完成" << endl;
		waitKey(100);



	}



