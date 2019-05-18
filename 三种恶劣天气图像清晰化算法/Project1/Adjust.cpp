#include "Adjust.h"
#include<math.h>
#include<opencv.hpp>
using namespace std;
using namespace cv;
#define random(x) (rand()%x)

//构造函数
Adjustment::Adjustment()
{
	histSize[0] = 256;
	histSize[1] = 256;
	histSize[2] = 256;
	hranges[0] = 0.0;
	hranges[1] = 256.0;
	ranges[0] = hranges;
	ranges[1] = hranges;
	ranges[2] = hranges;
	channels[0] = 0;
	channels[1] = 1;
	channels[2] = 2;
}


		/*合成一种算法*/
Mat Adjustment::de_all(cv::Mat  image, IplImage *Image1) {


	return de_sand(image, Image1, 2);


}

		/*去沙城暴，*/
	/*//getvalue2（）得到方差等信息，getaImg得到颜色自适应调整图，SVD_de_sand奇异值分解方法，getfinalImg()被淘汰//*/
//总方法//flag为0，代表使用SVD方法，flag为1代表使用defog_3方法
Mat Adjustment::de_sand(cv::Mat  image, IplImage *Image1,int flag ) {

	Mat final = getvalue2(image, Image1);
	//imshow("final颜色自适应", final);

	if(flag==1) final = defog_3(final, 0);

	if (flag == 2) final = ad_light(final);

	//imshow("fianl0", final);
	
	//final = ad_light(final);
	//final.convertTo(aimg, CV_32FC3);

	if (flag == 0) {
		//直方图均衡化
		cv::Mat final_2, final_1;
		final.copyTo(final_2);
		final.copyTo(final_1);

		vector <cv::Mat> splitBGR(final_1.channels());//定义向量存储各通道图片的向量
		cv::split(final_1, splitBGR);
		vector <cv::Mat> splitBGR_2(final_2.channels());//定义向量存储各通道图片的向量
		cv::split(final_2, splitBGR_2);
		//对final_2各个通道进行直方图均衡化
		for (int i = 0; i < final_2.channels(); ++i) {
			equalizeHist(splitBGR_2[i], splitBGR_2[i]);
		}
		//SVD图像增强
		for (int i = 0; i < final_2.channels(); ++i) {
			splitBGR[i] = SVD_de_sand(splitBGR[i], splitBGR_2[i]);//equalizeHist(splitBGR_2[i], splitBGR_2[i]);
		}
		merge(splitBGR, final);//合并通道
	}
	
	//cout << final;

	return final;
}
//以下方法，均在de_sand方法中被调用//
Mat Adjustment::getvalue2(const  cv::Mat  image, IplImage *Image1) {

	//int length = str.length;
	//cout << "length:"<< length << endl;
	//char p[]; //= str.c_str();
	//strcpy(p, str.c_str());
	//IplImage *Image1 = cvLoadImage(str.c_str(), 1);
	
	//IplImage *Image1 = cvCreateImage(cvSize(image.rows, image.cols), 8, 3);

	//Mat temp = image.clone();

	//Image1->imageData = (char *)temp.data;



	// Split the image into different channels
	Mat i;
	//image.copyTo(i);
	//cout << i;

	//imshow("i", i);

	image.convertTo(i, CV_32FC3);
	std::vector<cv::Mat> rgbChannels(3);
	split(i, rgbChannels);
	hist_b = rgbChannels[0];
	hist_g = rgbChannels[1];
	hist_r = rgbChannels[2];


	//得到总的像素点数
	 num = image.rows*image.cols;
	 r = image.rows;
	 c = image.cols;
	std::cout << "总像素点数：" << num << std::endl;



	//获取其中的随机数,并求估计值

	for (int i = 0; i < r; i++) {
		float * data_r = rgbChannels[2].ptr<float>(i);
		float * data_g = rgbChannels[1].ptr<float>(i);
		float * data_b = rgbChannels[0].ptr<float>(i);
		for (int j = 0; j < c; j++) {

			x[0] += data_r[j];
			y[0] += pow(data_r[j], 2);

			x[1] += data_g[j];
			y[1] += pow(data_g[j], 2);

			x[2] += data_b[j];
			y[2] += pow(data_b[j], 2);

			//hist_r.at<uchar>(x, y) = x[1] + s[0] * abs(x[0] - hist_r.at<uchar>(x, y));
		}
	}

	for (int w = 0; w < 3; w++) {
		x[w] = x[w] / num;
		y[w] = pow(y[w] / num - pow(x[w], 2), 0.5);
	}

	   

		std::cout << "三通道估值之：x[0]：" << x[0] << "x[1]:" << x[1] << "x[2]" << x[2] << std::endl;
		std::cout << "三通道估值之：y[0]：" << y[0] << "y[1]:" << y[1] << "y[2]" << y[2] << std::endl;

		//merge(rgbChannels, aImg);

		getaImg(Image1, rgbChannels);
		
		return aimg;

}
void Adjustment::getaImg(IplImage *Image1, std::vector<cv::Mat> rgbChannels) {

	cv::Mat aImg;
	//求取延伸系数a
	double MinValue;
	double MaxValue;

	CvPoint MinLocation;
	CvPoint MaxLocation;


	cvSetImageCOI(Image1, 1);
	cvMinMaxLoc(Image1, &MinValue, &MaxValue, &MinLocation, &MaxLocation);

	double a = (double)255 / (double)(MaxValue- MinValue);
	std::cout << "三通道估值之：a：" << a<<std::endl;
	cout << "MinLocation:" << MinLocation.x << "," << MinLocation.y << endl;


	//对各通道进行调整
	double s[3] = { 0 };
	for (int x = 0; x < 3; x++) 
		s[x] = a * y[x] / y[1];

	//s[1] = 1;

	std::cout << "三通道估值之：s[0]：" <<s[0] << "s[1]:" << s[1] << "s[2]" << s[2] << std::endl;
	
	double n = 0;
	for (int i = 0; i < r; i++) {
		float * data_r = rgbChannels[2].ptr<float>(i);
		float * data_g = rgbChannels[1].ptr<float>(i);
		float * data_b = rgbChannels[0].ptr<float>(i);
		for (int j = 0; j < c; j++) {
			
			n = x[1] - s[0] * (x[0] - data_r[j]);
			if (n < 0) {
				n = 0;
				//cout << "改变了一次" << endl;
			}
			if (n > 255) {
				n = 255;
				//cout << "改变成255" << endl;
			}
			data_r[j] = (float)n;

			n = x[1] - s[1] * (x[1] - data_g[j]);
			if (n< 0) n = 0;
			if (n > 255)n = 255;
			data_g[j] = (float)n;
			
			n = x[1] - s[2] * (x[2] - data_b[j]);
			if (n < 0) n = 0;
			if (n > 255)n = 255;
			data_b[j] = (float)n;
			
			//hist_r.at<uchar>(x, y) = x[1] + s[0] * abs(x[0] - hist_r.at<uchar>(x, y));
		}
	}


	

	//合并三通道
	 //rgbChannels[0]= hist_b ;
	 //rgbChannels[1]= hist_g ;
	 //rgbChannels[2]= hist_r ;
	//rgbChannels.push_back(hist_b);
	//rgbChannels.push_back(hist_g);
	//rgbChannels.push_back(hist_r);


	//for (int i = 0; i < aimg.channels(); ++i) {
		//equalizeHist(rgbChannels[i], rgbChannels[i]);
	//}
	merge(rgbChannels, aimg);//合并通道
	aimg.convertTo(aimg, CV_8UC3);
	//imshow("aimg", aimg);
	//aImg.copyTo(aimg);//复制图像

	//getfinalImg();

}
void Adjustment::getfinalImg() {


	//直方图均衡化
	cv::Mat aimg_2;
	vector <cv::Mat> splitBGR(aimg.channels());//定义向量存储各通道图片的向量
	cv::split(aimg, splitBGR);//对各个通道进行直方图均衡化

	for (int i = 0; i < aimg.channels(); ++i) {
		equalizeHist(splitBGR[i], splitBGR[i]);
	}
	merge(splitBGR, aimg_2);//合并通道


	

	//离散余弦变换
	cv::cvtColor(aimg, aimg, CV_BGR2GRAY, 0);

	//equalizeHist(aimg, aimg_2);

	//cv::resize(aimg, aimg, cv::Size(512, 512));
	aimg.convertTo(aimg, CV_32F, 1.0 / 255);
	cv::Mat aimg_DCT;
	cv::dct(aimg, aimg_DCT);

	cv::cvtColor(aimg_2, aimg_2, CV_BGR2GRAY, 0);
	//cv::resize(aimg, aimg, cv::Size(512, 512));
	aimg.convertTo(aimg_2, CV_32F, 1.0 / 255);
	cv::Mat aimg2_DCT;
	cv::dct(aimg, aimg2_DCT);

	//imshow("dct", aimg_2);
	

	//奇异值分解SVD
	aimg_DCT.convertTo(aimg_DCT, CV_32FC1);
	aimg2_DCT.convertTo(aimg2_DCT, CV_32FC1);

	cv::Mat U, W_1, V,W_2;

	cv::SVD::compute(aimg2_DCT, W_2, U, V);
    cv::SVD::compute(aimg_DCT, W_1, U, V);
	//U,V 进行处理


	//求比例系数
	cv::Mat w = Mat::zeros(Size(W_1.rows, W_1.rows), CV_32FC1);
	int len = W_1.rows;
	float max_1 = 0;
	float max_2 = 0;
	for (int i = 0; i < len; ++i) {

		if (W_1.ptr<float>(i)[0] == W_2.ptr<float>(i)[0])
			cout << "相同的奇异值：" << W_1.ptr<float>(i)[0] << endl;

		if (max_1 < W_1.ptr<float>(i)[0]) {
			max_1 = W_1.ptr<float>(i)[0];
			cout << "max_1更新一次：" << max_1<<endl;
		}
		if (max_2 < W_2.ptr<float>(i)[0]) {
			max_2 = W_2.ptr<float>(i)[0];
			cout << "max_2更新一次：" << max_2 << endl;
		}

	}
		

	float a = max_2/max_1;
	cout << "SVD中的比例系数：" << a << endl;


	//求新的w
	for (int i = 0; i < len; ++i) {
		w.ptr<float>(i)[i] = (a* W_1.ptr<float>(i)[0] + W_1.ptr<float>(i)[0] / a) / 2;
	}

	//求取最后的final矩阵
	finalImg = U * w *V;
	

	//反余弦变换IDCT
	idct(finalImg, finalImg);
	//finalImg.convertTo(finalImg, CV_8UC1);
	//cvtColor(finalImg, finalImg, CV_GRAY2BGR, 1);
	
	

	imshow("fianl", finalImg);

	
	//cout << U << endl;
	//cout << W << endl;
	//cout << V << endl;



}
Mat Adjustment::SVD_de_sand(Mat sec,Mat sec_hist) {

	/*离散余弦变换*/
	sec.convertTo(sec, CV_32F, 1.0 / 255);
	cv::Mat sec_DCT;
	cv::dct(sec, sec_DCT);

	sec_hist.convertTo(sec_hist, CV_32F, 1.0 / 255);
	cv::Mat sec_hist_DCT;
	cv::dct(sec_hist, sec_hist_DCT);

	//imshow("sec_DCT", sec_DCT);
	//imshow("sec_hist_DCT", sec_hist_DCT);


	/*奇异值分解*/
	sec_DCT.convertTo(sec_DCT, CV_32FC1);
	sec_hist_DCT.convertTo(sec_hist_DCT, CV_32FC1);

	cv::Mat U, W_1, V, W_2;

	cv::SVD::compute(sec_hist_DCT, W_2, U, V);
	cv::SVD::compute(sec_DCT, W_1, U, V);


	/*求比例系数*/
	cv::Mat w = Mat::zeros(Size(W_1.rows, W_1.rows), CV_32FC1);
	int len = W_1.rows;
	float max_1 = 0;
	float max_2 = 0;

	//max_1 = W_1.ptr<float>(0)[0];
	//max_2 = W_2.ptr<float>(0)[0];

	for (int i = 0; i < 1; ++i) {
		if (W_1.ptr<float>(i)[0] == W_2.ptr<float>(i)[0])
			cout << "相同的奇异值：" << W_1.ptr<float>(i)[0] << endl;
		if (max_1 < W_1.ptr<float>(i)[0]) {
			max_1 = W_1.ptr<float>(i)[0];
			cout << "max_1更新一次：" << max_1 << endl;
		}
		if (max_2 < W_2.ptr<float>(i)[0]) {
			max_2 = W_2.ptr<float>(i)[0];
			cout << "max_2更新一次：" << max_2 << endl;
		}
	}
	
	float a = max_2 /max_1 ;
	cout << "SVD中的比例系数：" << a << endl;


	/*求新的w*/
	float x;
	for (int i = 0; i < len; ++i) {
		x = (a* W_1.ptr<float>(i)[0] + W_2.ptr<float>(i)[0] / a) / 2;
	
		//if (x < 0) x = 0;
		//if (x > 255) x = 255;
			 w.ptr<float>(i)[i] = x;
	}
	//求取最后的final矩阵
	//pow(U,1.1,U);
	//pow(V, 1.1, V);

	Mat final = U * w *V;
	//反余弦变换IDCT
	idct(final, final);

	for (int i = 0; i < final.rows; i++) {
		float * data = final.ptr<float>(i);
		for (int j = 0; j < final.cols; j++) {
			//cout << data[j] << "**";
			if (data[j] < 0) data[j] = 0;
			if (data[j] > 1) data[j] = 1;
		}
	}
	//finalImg.convertTo(finalImg, CV_8UC1);
	//cvtColor(finalImg, finalImg, CV_GRAY2BGR, 1);


	//imshow("fianlSVD", final);
	return final;
}


		/*去雾算法，共有三种，defog_2不推荐使用*/
//暗通道去雾算法//flag为0，代表要直方图均衡化，否则不直方图均衡化
Mat Adjustment::defog( Mat image ,int flag ) {

	CV_Assert(!image.empty() && image.channels() == 3);
	//图片的归一化  
	Mat fImage;
	image.convertTo(fImage, CV_32FC3, 1.0 / 255, 0);
	//规定patch的大小,且均为奇数  
	int hPatch = 15;//15;
	int vPatch = 15;// 15;
	//给归一化的图片添加边界  
	Mat fImageBorder;
	copyMakeBorder(fImage, fImageBorder, vPatch / 2, vPatch / 2, hPatch / 2, hPatch / 2, BORDER_REPLICATE);
	//分离通道  
	vector<Mat> fImageBorderVector(3);
	split(fImageBorder, fImageBorderVector);
	//创建darkChannel  
	Mat darkChannel(image.rows, image.cols, CV_32FC1);
	double minTemp, minPixel;
	//根据darkChannel的定义  
	for (unsigned int r = 0; r < darkChannel.rows; r++)
	{
		for (unsigned int c = 0; c < darkChannel.cols; c++)
		{
			minPixel = 1.0;
			for (vector<Mat>::iterator it = fImageBorderVector.begin(); it != fImageBorderVector.end(); it++)
			{
				Mat roi(*it, Rect(c, r, hPatch, vPatch));
				minMaxLoc(roi, &minTemp);
				minPixel = min(minPixel, minTemp);
			}
			darkChannel.at<float>(r, c) = float(minPixel);
		}
	}
	//namedWindow("darkChannel", 1);
	//imshow("darkChannel", darkChannel);


	/*第2步：求出 A(global atmospheric light)*/
//2.1 计算出darkChannel中,前top个亮的值,论文中取值为0.1%  
	float top = 0.001;
	float numberTop = top * darkChannel.rows*darkChannel.cols;
	Mat darkChannelVector;
	darkChannelVector = darkChannel.reshape(1, 1);
	Mat_<int> darkChannelVectorIndex;
	sortIdx(darkChannelVector, darkChannelVectorIndex, CV_SORT_EVERY_ROW + CV_SORT_DESCENDING);
	//制作掩码  
	Mat mask(darkChannelVectorIndex.rows, darkChannelVectorIndex.cols, CV_8UC1);//注意mask的类型必须是CV_8UC1  
	for (unsigned int r = 0; r < darkChannelVectorIndex.rows; r++)
	{
		for (unsigned int c = 0; c < darkChannelVectorIndex.cols; c++)
		{
			if (darkChannelVectorIndex.at<int>(r, c) <= numberTop)
				mask.at<uchar>(r, c) = 1;
			else
				mask.at<uchar>(r, c) = 0;
		}
	}
	Mat darkChannelIndex = mask.reshape(1, darkChannel.rows);
	vector<double> A(3);//分别存取A_b,A_g,A_r  
	vector<double>::iterator itA = A.begin();
	vector<Mat>::iterator it = fImageBorderVector.begin();
	//2.2在求第三步的t(x)时，会用到以下的矩阵，这里可以提前求出  
	vector<Mat> fImageBorderVectorA(3);
	vector<Mat>::iterator itAA = fImageBorderVectorA.begin();
	for (; it != fImageBorderVector.end() && itA != A.end() && itAA != fImageBorderVectorA.end(); it++, itA++, itAA++)
	{
		Mat roi(*it, Rect(hPatch / 2, vPatch / 2, darkChannel.cols, darkChannel.rows));
		minMaxLoc(roi, 0, &(*itA), 0, 0, darkChannelIndex);//  
		(*itAA) = (*it) / (*itA); //[注意：这个地方有除号，但是没有判断是否等于0]  
	}


	/*第三步：求t(x)*/
		Mat darkChannelA(darkChannel.rows, darkChannel.cols, CV_32FC1);
		float omega = 0.7;// 0.95;//0<w<=1,论文中取值为0.95  
	//代码和求darkChannel的时候,代码差不多  
	for (unsigned int r = 0; r < darkChannel.rows; r++)
	{
		for (unsigned int c = 0; c < darkChannel.cols; c++)
		{
			minPixel = 1.0;
			for (itAA = fImageBorderVectorA.begin(); itAA != fImageBorderVectorA.end(); itAA++)
			{
				Mat roi(*itAA, Rect(c, r, hPatch, vPatch));
				minMaxLoc(roi, &minTemp);
				minPixel = min(minPixel, minTemp);
			}
			darkChannelA.at<float>(r, c) = float(minPixel);
		}
	}
	Mat tx = 1.0 - omega * darkChannelA;

	/*第四步：我们可以求J(x)*/
		float t0 = 0.1;//论文中取t0 = 0.1  
	Mat jx(image.rows, image.cols, CV_32FC3);
	for (size_t r = 0; r < jx.rows; r++)
	{
		for (size_t c = 0; c < jx.cols; c++)
		{
			jx.at<Vec3f>(r, c) = Vec3f((fImage.at<Vec3f>(r, c)[0] - A[0]) / max(tx.at<float>(r, c), t0) + A[0], (fImage.at<Vec3f>(r, c)[1] - A[1]) / max(tx.at<float>(r, c), t0) + A[1], (fImage.at<Vec3f>(r, c)[2] - A[2]) / max(tx.at<float>(r, c), t0) + A[2]);
		}
	}

	if (flag == 0) {
		jx.convertTo(jx, CV_8UC3, 255, 0);

		normalize(jx, jx, 0, 255, CV_MINMAX);
		//jx.convertTo(jx, CV_32FC3, 255, 0);
		vector <cv::Mat> splitBGR(jx.channels());
		cv::split(jx, splitBGR);
		//对final_2各个通道进行直方图均衡化
		for (int i = 0; i < jx.channels(); ++i) {
			equalizeHist(splitBGR[i], splitBGR[i]);
		}
		merge(splitBGR, jx);

		jx = ad_light_Log(jx);
		jx = drop_light(jx);


		
	}
	jx.convertTo(jx, CV_8UC3);

	return jx;

}
//第二类去雾算法//
Mat Adjustment::defog_2(Mat deFog1) {
	cv::Mat channels[3];    
	split(deFog1, channels);
	//不知道什么原因vector无法使用 只能用数组来表示   
	for(int c=0;c<3;c++)     
		channels[c]= grayStretch(channels[c],0.001,0.5); 
	//根据实验 暗色像素的比例应该设置的较小效果会比较好   
	merge(channels,3,deFog1);

	return deFog1;
}
Mat Adjustment::grayStretch(cv::Mat src, double lowcut, double highcut)//defog_2之内会使用的方法
{
	//[1]--统计各通道的直方图   
	//参数   
	const int bins = 256;
	int hist_size = bins;
	float range[] = { 0,255 };
	const float* ranges[] = { range };
	MatND desHist;    int channels = 0;
	//计算直方图   
	calcHist(&src, 1, &channels, Mat(), desHist, 1, &hist_size, ranges, true, false);
	//[1]   
	//[2] --计算上下阈值   
	int pixelAmount = src.rows*src.cols;
	//像素总数  
	float Sum = 0;
	int minValue, maxValue;
	//求最小值   
	for (int i = 0; i < bins; i++) {
		Sum = Sum + desHist.at<float>(i);
		if (Sum >= pixelAmount * lowcut*0.01)
		{
			minValue = i;
			//qDebug() << "minValue" << minValue;
			break;
		}
	}
	//求最大值    
	Sum = 0;
	for (int i = bins - 1; i >= 0; i--) {
		Sum = Sum + desHist.at<float>(i);
		if (Sum >= pixelAmount * highcut*0.01)
		{
			maxValue = i;
			//qDebug() << "maxValue" << maxValue;
			break;
		}
	}
	//[2] 
	//[3]--对各个通道进行线性拉伸   
	Mat dst = src;
	//判定是否需要拉伸   
	if (minValue > maxValue)
		return src;
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++) {
			if (src.at<uchar>(i, j) < minValue)
				dst.at<uchar>(i, j) = 0;
			if (src.at<uchar>(i, j) > maxValue)
				dst.at<uchar>(i, j) = 255;
			else {
				//注意这里做除法要使用double类型      
				double pixelValue = ((src.at<uchar>(i, j) - minValue) /
					(double)(maxValue - minValue)) * 255;
				dst.at<uchar>(i, j) = (int)pixelValue;
			}
		}    //[3]

	return dst;
}

/*
//卷积滤波去雾算法//flag为0，代表要直方图均衡化，否则不直方图均衡化
Mat Adjustment::defog_3(Mat image,int flag) {

	//归一化
	image.convertTo(image, CV_32FC3, 1.0 / 255, 0);

	//定义参数
	int alf = 5;//标准差
	int w = 6 * alf + 1;
	int n = floor((w + 1) / 2);

	int M = image.rows;//行
	int N = image.cols;//列

	Mat image_2;
	image.copyTo(image_2);
	float sum = 0;
	Mat F(w, w, CV_32FC1);

	//高斯函数
	for (int i = 0; i < w; i++) {
		float *data = F.ptr<float>(i);
		for (int j = 0; j < w; j++) {
			data[j] = exp(-(pow((i - n) , 2) +pow( (j - n) ,2)) / pow(alf ,2));
			sum += data[j];
		}
	}
	for (int i = 0; i < w; i++) {
		float *data = F.ptr<float>(i);
		for (int j = 0; j < w; j++) {
			data[j] /= sum;//归一化
		}
	}


	//卷积滤波
	filter2D(image_2, image_2, -1, F);
	//求像素均值
	float sum_r = 0;
	float sum_b = 0;
	float sum_g = 0;
	float Fr_mean = 0;
	float Fb_mean = 0;
	float Fg_mean = 0;
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++)
		{
			sum_b += image_2.at<Vec3f>(i, j)[0];
			sum_g += image_2.at<Vec3f>(i, j)[1];
			sum_r += image_2.at<Vec3f>(i, j)[2];
		}
	}
	Fr_mean = sum_r / (M*N);
	Fb_mean = sum_b / (M*N);
	Fg_mean = sum_g / (M*N);
	cout << "Fr_mean:" << Fr_mean << "Fb_mean:" << Fb_mean << "Fg_mean:" << Fg_mean << endl;


	

	//像素处理
	//image.convertTo(image, CV_32FC3, 255, 0);
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++)
		{
			image_2.at<Vec3f>(i, j)[0] = exp(log(image.at<Vec3f>(i, j)[0] + 1) - log(Fb_mean + 1));
			image_2.at<Vec3f>(i, j)[1] = exp(log(image.at<Vec3f>(i, j)[1] + 1) - log(Fg_mean + 1));
			image_2.at<Vec3f>(i, j)[2] = exp(log(image.at<Vec3f>(i, j)[2] + 1) - log(Fr_mean + 1));
		}
	}
	vector <cv::Mat> splitBGR(image_2.channels());//定义向量存储各通道图片的向量
	cv::split(image_2, splitBGR);
	double maxvalue = 0;
	double maxV, minV;
	Point maxP, minP;
	for (int i = 0; i < 3; i++) {
	minMaxLoc(splitBGR[0], &minV, &maxV, &minP, &maxP);
	if(maxvalue < maxV)maxvalue = maxV;
	}
	cout <<  "maxvalue:" << maxvalue << endl;

	//变换到0——255区间内
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++)
		{
			image_2.at<Vec3f>(i, j)[0] =((image_2.at<Vec3f>(i, j)[0] ) / maxvalue);
			//image_2.at<Vec3f>(i, j)[0]--;
			image_2.at<Vec3f>(i, j)[1] = ((image_2.at<Vec3f>(i, j)[1] ) / maxvalue);
			//image_2.at<Vec3f>(i, j)[1]--;
			image_2.at<Vec3f>(i, j)[2] = ((image_2.at<Vec3f>(i, j)[2] ) / maxvalue);
			//image_2.at<Vec3f>(i, j)[2]--;
		}
	}
	
	image_2.convertTo(image_2, CV_8UC3,255,0);
	//cout << image_2;
	
	cv::split(image_2, splitBGR);
	for (int i = 0; i < 3; i++) {
		splitBGR[i]=Adaptive_contrast(splitBGR[i]);
	}

	//直方图均衡化
	if (flag == 0) {
		//对final_2各个通道进行直方图均衡化
		for (int i = 0; i < 3; ++i) {
			equalizeHist(splitBGR[i], splitBGR[i]);
		}
	}
		merge(splitBGR, image_2);//合并通道

		image_2 = ad_light_Log(image_2);
		image_2 = drop_light(image_2);

		//image_2.convertTo(image_2, CV_8UC3);

		Mat i;
		medianBlur(image_2,i, 1);
		//bilateralFilter(image_2, i, 5, 5 * 2, 5 / 2);



		return i;// mage_2;
}
Mat Adjustment::Adaptive_contrast(Mat image) //defog_3会使用到的方法：灰度拉伸
{
	int a0 = 0;
	int b0 = 240;// 255;
	float Th = 0.03;// 0.02;
	float total[256] = { 0 };
	float jtem[256] = { 0 };
	float jtotal[256] = { 0 };
	int sum=0;

	int M = image.rows;
	int N = image.cols;

	for (int i = 0; i < M; i++) {
		uchar * data = image.ptr<uchar>(i);
		for (int j = 0; j < N; j++) {
			total[data[j]]++;
		}
	}
	for (int i = 1; i < 256; i++) {
		sum += total[i];
		jtem[i] = sum;
	}

	int Tlow = 0;
	int Thigh = 0;
	for (int i = 1; i < 256; i++) {
		jtotal[i] = jtem[i] / sum;
		if ((jtotal[i] >= Th) && (jtotal[i] > jtotal[i - 1]))
		{
			Tlow = i - 1;
			break;
		}
	}
	for (int i = 1; i < 256; i++) {
		jtotal[i] = jtem[i] / sum;
		if ((jtotal[i] >= 1-Th) && (jtotal[i] > jtotal[i - 1]))
		{
			Thigh = i - 1;
			break;
		}
	}

	image.convertTo(image, CV_32F);
	for (int i = 0; i < M; i++) {
		float * data = image.ptr<float>(i);
		for (int j = 0; j < N; j++) {
			data[j] = a0 + (data[j] - Tlow)*(b0 - a0) / (Thigh - Tlow);
		}
	}
	image.convertTo(image, CV_8U);

	return image;

}
*/


//卷积滤波去雾算法//flag为0，代表要直方图均衡化，否则不直方图均衡化
Mat Adjustment::defog_3(Mat image, int flag) {

	Mat image_3;
	image.copyTo(image_3);

	//归一化
	double minv = 0.0, maxv = 0.0;
	double* minp = &minv;
	double* maxp = &maxv;
	minMaxIdx(image, minp, maxp);

	image.convertTo(image, CV_32FC3, 1.0 / maxv, 0);       //image.convertTo(image, CV_32FC3, 1.0 / 255, 0);

	//定义参数
	int alf = 5;//标准差
	int w = 6 * alf + 1;
	int n = floor((w + 1) / 2);

	int M = image.rows;//行
	int N = image.cols;//列

	Mat image_2;
	image.copyTo(image_2);
	float sum = 0;
	Mat F(w, w, CV_32FC1);

	//高斯函数
	for (int i = 1; i <= w; i++) {
		float *data = F.ptr<float>(i - 1);
		for (int j = 1; j <= w; j++) {
			data[j - 1] = exp(float(-((i - n)*(i - n) + (j - n)*(j - n))) / (alf*alf));
			sum += data[j - 1];
		}
	}
	for (int i = 0; i < w; i++) {
		float *data = F.ptr<float>(i);
		for (int j = 0; j < w; j++) {
			data[j] /= sum;//归一化
		}
	}

	//卷积滤波
	filter2D(image_2, image_2, -1, F);
	//求像素均值
	float sum_r = 0;
	float sum_b = 0;
	float sum_g = 0;
	float Fr_mean = 0;
	float Fb_mean = 0;
	float Fg_mean = 0;
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++)
		{
			sum_b += image_2.at<Vec3f>(i, j)[0];
			sum_g += image_2.at<Vec3f>(i, j)[1];
			sum_r += image_2.at<Vec3f>(i, j)[2];
		}
	}
	Fr_mean = sum_r / (M*N);
	Fb_mean = sum_b / (M*N);
	Fg_mean = sum_g / (M*N);

	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++)
		{
			image_2.at<Vec3f>(i, j)[0] = exp(log(image_3.at<Vec3b>(i, j)[0] + 1) - log(255 * Fb_mean + 1));
			image_2.at<Vec3f>(i, j)[1] = exp(log(image_3.at<Vec3b>(i, j)[1] + 1) - log(255 * Fg_mean + 1));
			image_2.at<Vec3f>(i, j)[2] = exp(log(image_3.at<Vec3b>(i, j)[2] + 1) - log(255 * Fr_mean + 1));
		}
	}

	double minvx = 0.0, maxvx = 0.0;
	double* minpx = &minvx;
	double* maxpx = &maxvx;
	minMaxIdx(image_2, minpx, maxpx);

	double maxvalue = 0;

	maxvalue = maxvx;

	//变换到0——255区间内
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++)
		{
			image_2.at<Vec3f>(i, j)[0] = ((image_2.at<Vec3f>(i, j)[0]) / maxvalue);
			//image_2.at<Vec3f>(i, j)[0]--;
			image_2.at<Vec3f>(i, j)[1] = ((image_2.at<Vec3f>(i, j)[1]) / maxvalue);
			//image_2.at<Vec3f>(i, j)[1]--;
			image_2.at<Vec3f>(i, j)[2] = ((image_2.at<Vec3f>(i, j)[2]) / maxvalue);
			//image_2.at<Vec3f>(i, j)[2]--;
		}
	}

	image_2.convertTo(image_2, CV_8UC3, 255, 0);
	//cout << image_2;

	image_2 = Adaptive_contrast(image_2);

	return image_2;
}
Mat Adjustment::Adaptive_contrast(Mat image) //defog_3会使用到的方法：灰度拉伸
{
	int a0 = 0;
	int b0 = 255;
	float Th = 0.02;
	float total[256] = { 0 };
	float jtem[256] = { 0 };
	float jtotal[256] = { 0 };
	int sum = 0;

	int M = image.rows;
	int N = image.cols;

	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			int point_total = image.at<Vec3b>(i, j)[2];
			total[point_total]++;
		}
	}

	for (int i = 1; i < 256; i++) {
		sum += total[i];
		jtem[i] = sum;
	}

	int Tlow = 0;
	int Thigh = 0;
	for (int i = 1; i < 256; i++) {
		jtotal[i] = jtem[i] / sum;
		if ((jtotal[i] >= Th) && (jtotal[i] > jtotal[i - 1]))
		{
			Tlow = i - 1;
			break;
		}
	}

	for (int i = 1; i < 256; i++) {
		jtotal[i] = jtem[i] / sum;
		if ((jtotal[i] >= 1 - Th) && (jtotal[i] > jtotal[i - 1]))
		{
			Thigh = i - 1;
			break;
		}
	}

	image.convertTo(image, CV_32F);

	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			image.at<Vec3f>(i, j)[0] = a0 + (image.at<Vec3f>(i, j)[0] - Tlow - 1)*(b0 - a0) / (Thigh - Tlow);
			image.at<Vec3f>(i, j)[1] = a0 + (image.at<Vec3f>(i, j)[1] - Tlow - 1)*(b0 - a0) / (Thigh - Tlow);
			image.at<Vec3f>(i, j)[2] = a0 + (image.at<Vec3f>(i, j)[2] - Tlow - 1)*(b0 - a0) / (Thigh - Tlow);
		}
	}

	image.convertTo(image, CV_8U);

	return image;
}



Mat Adjustment::defog_4(Mat image, int flag) {
	//归一化
	image.convertTo(image, CV_32FC3, 1.0 / 255, 0);

	//定义参数
	int alf = 5;//标准差
	int w = 6 * alf + 1;
	int n = floor((w + 1) / 2);

	int M = image.rows;//行
	int N = image.cols;//列

	Mat image_2;
	image.copyTo(image_2);
	float sum = 0;
	Mat F(w, w, CV_32FC1);

	//高斯函数
	for (int i = 0; i < w; i++) {
		float *data = F.ptr<float>(i);
		for (int j = 0; j < w; j++) {
			data[j] = exp(-(pow((i - n), 2) + pow((j - n), 2)) / pow(alf, 2));
			sum += data[j];
		}
	}
	for (int i = 0; i < w; i++) {
		float *data = F.ptr<float>(i);
		for (int j = 0; j < w; j++) {
			data[j] /= sum;//归一化
		}
	}


	//卷积滤波
	filter2D(image_2, image_2, -1, F);
	//求像素均值
	float sum_r = 0;
	float sum_b = 0;
	float sum_g = 0;
	float Fr_mean = 0;
	float Fb_mean = 0;
	float Fg_mean = 0;
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++)
		{
			sum_b += image_2.at<Vec3f>(i, j)[0];
			sum_g += image_2.at<Vec3f>(i, j)[1];
			sum_r += image_2.at<Vec3f>(i, j)[2];
		}
	}
	Fr_mean = sum_r / (M*N);
	Fb_mean = sum_b / (M*N);
	Fg_mean = sum_g / (M*N);
	cout << "Fr_mean:" << Fr_mean << "Fb_mean:" << Fb_mean << "Fg_mean:" << Fg_mean << endl;


	Mat L;
	image_2.copyTo(L);
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++)
		{
			L.at<Vec3f>(i, j)[0] =255 - L.at<Vec3f>(i, j)[0] * Fb_mean;
			L.at<Vec3f>(i, j)[1] =255 - L.at<Vec3f>(i, j)[1] * Fg_mean;
			L.at<Vec3f>(i, j)[2] =255 - L.at<Vec3f>(i, j)[2] * Fr_mean;
		}
	}
	L.convertTo(L, CV_BGR2YCrCb,1.0/255);
	//L.convertTo(L, CV_32F);
	vector <cv::Mat> splitBGR_L(L.channels());
	//L = splitBGR_L[0];
	//splitBGR_L[1] = splitBGR_L[0];
	//splitBGR_L[2] = splitBGR_L[0];
	//merge(L, splitBGR_L);

	//imshow("L__CV_BGR2YCrCb", L);

	//L.convertTo(L, CV_32F);



	//像素处理
	vector <cv::Mat> splitBGR(image_2.channels());//定义向量存储各通道图片的向量
	cv::split(image_2, splitBGR);
	for (int i = 0; i < r; i++) {
		float * data = splitBGR_L[0].ptr<float>(i);
		float * data_r = splitBGR[2].ptr<float>(i);
		float * data_g = splitBGR[1].ptr<float>(i);
		float * data_b = splitBGR[0].ptr<float>(i);
		for (int j = 0; j < c; j++) {
			data_r[j] = exp(log(data_r[j] + 1) - log(data[j] + 1));
			data_g[j] = exp(log(data_g[j] + 1) - log(data[j] + 1));
			data_b[j] = exp(log(data_b[j] + 1) - log(data[j] + 1));
		}
	}

	/*
	//image.convertTo(image, CV_32FC3, 255, 0);
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++)
		{
			image_2.at<Vec3f>(i, j)[0] = exp(log(image.at<Vec3f>(i, j)[0] + 1) - log(L.at<Vec3f>(i, j)[0] + 1));
			image_2.at<Vec3f>(i, j)[1] = exp(log(image.at<Vec3f>(i, j)[1] + 1) - log(L.at<Vec3f>(i, j)[0] + 1));
			image_2.at<Vec3f>(i, j)[2] = exp(log(image.at<Vec3f>(i, j)[2] + 1) - log(L.at<Vec3f>(i, j)[0] + 1));
		}
	}
	*/

	double maxvalue = 0;
	double maxV, minV;
	Point maxP, minP;
	for (int i = 0; i < 3; i++) {
		minMaxLoc(splitBGR[0], &minV, &maxV, &minP, &maxP);
		if (maxvalue < maxV)maxvalue = maxV;
	}
	cout << "maxvalue:" << maxvalue << endl;

	//变换到0——255区间内
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++)
		{
			image_2.at<Vec3f>(i, j)[0] = ((image_2.at<Vec3f>(i, j)[0]) / maxvalue);
			//image_2.at<Vec3f>(i, j)[0]--;
			image_2.at<Vec3f>(i, j)[1] = ((image_2.at<Vec3f>(i, j)[1]) / maxvalue);
			//image_2.at<Vec3f>(i, j)[1]--;
			image_2.at<Vec3f>(i, j)[2] = ((image_2.at<Vec3f>(i, j)[2]) / maxvalue);
			//image_2.at<Vec3f>(i, j)[2]--;
		}
	}

	image_2.convertTo(image_2, CV_8UC3, 255, 0);
	//cout << image_2;

	cv::split(image_2, splitBGR);
	for (int i = 0; i < 3; i++) {
		splitBGR[i] = Adaptive_contrast(splitBGR[i]);
	}

	//直方图均衡化
	if (flag == 0) {
		//对final_2各个通道进行直方图均衡化
		for (int i = 0; i < 3; ++i) {
			equalizeHist(splitBGR[i], splitBGR[i]);
		}
	}
	merge(splitBGR, image_2);//合并通道

	//image_2 = ad_light_Log(image_2);
	//image_2 = drop_light(image_2);

	//image_2.convertTo(image_2, CV_8UC3);

	//Mat i;
	//medianBlur(image_2, i, 1);
	//bilateralFilter(image_2, i, 5, 5 * 2, 5 / 2);



	return image_2;
}

		/*低亮度增强，其中使用了defog_3的方法*/
Mat Adjustment::ad_light(Mat image) {

	
	uchar * data_r, *data_g, *data_b;

	//得到总的像素点数
	num = image.rows*image.cols;
	r = image.rows;
	c = image.cols;
	std::cout << "总像素点数：" << num << std::endl;

	std::vector<cv::Mat> rgbChannels(3);
	split(image, rgbChannels);
	//转换成对应像素
	for (int i = 0; i < r; i++) {
		  data_r = rgbChannels[2].ptr<uchar>(i);
		 data_g = rgbChannels[1].ptr<uchar>(i);
		 data_b = rgbChannels[0].ptr<uchar>(i);
		for (int j = 0; j < c; j++) {

			data_r[j] = (uchar)255- data_r[j];
			data_g[j] = (uchar)255 - data_g[j];
			data_b[j] = (uchar)255 - data_b[j];
		}
	}

	merge(rgbChannels,image);

	//namedWindow("转换为雾图image", 1);
	//imshow("转换为雾图image", image);



	Mat jx = defog_3(image,1);

	jx.convertTo(jx, CV_32FC3, 1.0 / 255, 0);
	//defog(image).copyTo(img);


	Vec3f point = Vec3f(1, 1, 1);

	for (size_t r = 0; r < jx.rows; r++)
	{
		for (size_t c = 0; c < jx.cols; c++)
		{
			jx.at<Vec3f>(r, c) = point - jx.at<Vec3f>(r, c);
		}
	}

	//cout << img << endl;
	//转换成对应像素
	jx.convertTo(jx, CV_8UC3, 255, 0);

	//jx = ad_light_Log(jx);
	//jx = drop_light(jx);

	jx.convertTo(jx, CV_32FC3, 1.0 / 255, 0);

	//namedWindow("jx", 1);
	//imshow("jx", jx);

	return jx;
	

}

		/*对数方法低亮度增强，两者联合使用*/
Mat Adjustment::ad_light_Log(Mat img) {

	Mat imageLog(img.size(), CV_32FC3);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++)
		{
			imageLog.at<Vec3f>(i, j)[0] = log(1 + img.at<Vec3b>(i, j)[0]);
			imageLog.at<Vec3f>(i, j)[1] = log(1 + img.at<Vec3b>(i, j)[1]);
			imageLog.at<Vec3f>(i, j)[2] = log(1 + img.at<Vec3b>(i, j)[2]);
		}
	}
	//归一化到0~255  	
	normalize(imageLog, imageLog, 0, 255, CV_MINMAX);
	//转换成8bit图像显示  	
	convertScaleAbs(imageLog, imageLog);
	//imshow("Soure", img);
	//imshow("after", imageLog);

	return imageLog;
	
}
Mat Adjustment::drop_light(Mat image) {

	Mat imageGamma(image.size(), CV_32FC3);
	for (int i = 0; i < image.rows; i++) { 
		for (int j = 0; j < image.cols; j++) 
		{ imageGamma.at<Vec3f>(i, j)[0] = (image.at<Vec3b>(i, j)[0])*(image.at<Vec3b>(i, j)[0])*(image.at<Vec3b>(i, j)[0]);	
		imageGamma.at<Vec3f>(i, j)[1] = (image.at<Vec3b>(i, j)[1])*(image.at<Vec3b>(i, j)[1])*(image.at<Vec3b>(i, j)[1]);	
		imageGamma.at<Vec3f>(i, j)[2] = (image.at<Vec3b>(i, j)[2])*(image.at<Vec3b>(i, j)[2])*(image.at<Vec3b>(i, j)[2]); 
		} 
	}	
	//归一化到0~255  
	normalize(imageGamma, imageGamma, 0, 255, CV_MINMAX);	
	//转换成8bit图像显示  	
	convertScaleAbs(imageGamma, imageGamma);

	return imageGamma;
	
}