#include <iostream>
#include <cv.h>
#include <vector>
#include "opencv2/opencv.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

// #define VIEW_WIDTH 700
// #define VIEW_HEIGHT 560

using namespace std;
using namespace cv;

void transform(Point2f* src_vertices, Point2f* dst_vertices, Mat& src, Mat &dst)
{
    Mat M = getPerspectiveTransform(src_vertices, dst_vertices);
    warpPerspective(src, dst, M, dst.size(), INTER_LINEAR, BORDER_CONSTANT);
}

int main()
{
    VideoCapture video("project_video.mp4");

    if(!video.isOpened())
        printf("Failed to open the video\n");

    Mat src, dst;
    Mat gray_img;
    Mat binary_img;
    
    Mat hls_img;
    Mat hls_plane;
    
    Mat gau;

    Point2f src_vertices[4]; // Point2f형은 (x, y)좌표로.

    video >> src;

    src_vertices[0] = Point(src.cols / 2 - 150, 500); // 좌상
    src_vertices[1] = Point(src.cols / 2 + 150, 500); // 우상
    src_vertices[2] = Point(src.cols / 2 + 600, 700); // 우하
    src_vertices[3] = Point(src.cols / 2 - 500, 700); // 좌하

    Point2f dst_vertices[4];
	
    dst_vertices[0] = Point(0, 0);     // 좌상
    dst_vertices[1] = Point(450, 0);   // 우상
    dst_vertices[2] = Point(450, 700); // 우하
    dst_vertices[3] = Point(0, 700);   // 좌하

    while(1)
    {
    	video >> src;

        /* 0. Bird eyes view Transform */
    	Mat M = getPerspectiveTransform(src_vertices, dst_vertices);
    	Mat dst(660, 500, CV_8UC3);

		/* src: 입력, dst: 저장할 변수, M: 변환 행렬 */
		warpPerspective(src, dst, M, dst.size(), INTER_LINEAR, BORDER_CONSTANT);

		/* 1. convert original image to HSL */
		cvtColor(dst, hls_img, COLOR_BGR2HLS);

		/* 2. Isolate Y and W from HLS image */
		Mat wImgMask, yImgMask, imgMask;
		// W
		inRange(hls_img, Scalar(0, 200, 0), Scalar(255, 255, 255), wImgMask);
		// Y
		inRange(hls_img, Scalar(10, 40, 115), Scalar(50, 200, 255), yImgMask);
		
		/* 3. Combine W and Y / Combine Original Img and HLS Img */
		imgMask = wImgMask | yImgMask;
		
		Mat imgResult;
		bitwise_and(dst, dst, imgResult, imgMask);

		/* 4. grayscale */
		cvtColor(imgResult, gray_img, COLOR_BGR2GRAY);
		
		/* 5. 가우시안 블러 */
		GaussianBlur(gray_img, gau, Size(11, 11), 0);
		
		/* 6. 캐니 에지 */
		Mat cannyImg;
		Canny(gau, cannyImg, 100, 150);
		
		/* 7. 좌우 차선 분리하기 (좌우 각각 다른 벡터에 저장되도록) */
		Mat leftROI, rightROI;
		// L 	
		Rect rect1(0, 0, 250, 660); // X, Y, W, H
		leftROI = cannyImg(rect1);
		// R
		Rect rect2(250, 0, 250, 660);
		rightROI = cannyImg(rect2);
		
		/* 8. 허프라인변환 */
	/*
		if (abs(slope) > slope_threshold) 
		{
			slopes.push_back(slope);
			new_lines.push_back(line);
		}
	*/
		
		vector<Vec4i> leftLines;
		vector<Vec4i> rightLines;
		
		// 시그마값 저장할 변수 선언
		double leftSigXY = 0;
		double leftSigX = 0;
		double leftSigY = 0;
		double leftSigXX = 0;	
		
		double rightSigXY = 0;
		double rightSigX = 0;
		double rightSigY = 0;
		double rightSigXX = 0;
		
		// 좌우 a0, a1 변수 선언
		double lefta0 = 0;
		double lefta1 = 0;
		double righta0 = 0;
		double righta1 = 0;
		
    	HoughLinesP(leftROI, leftLines, 1, CV_PI / 180, 50, 10, 20);
    	HoughLinesP(rightROI, rightLines, 1, CV_PI / 180, 20, 10, 10);
    	
    	// 좌우 n 값 구하기 
		double leftn = leftLines.size() * 2;
		double rightn = rightLines.size() * 2;
    	
    	// 이미지 자르기
    	Mat imgL = dst(Range(0, 660), Range(0, 250));
    	Mat imgR = dst(Range(0, 660), Range(250, 500));
    	
    	// 왼쪽 시그마값 구하기
    	for (int i = 0; i < leftLines.size(); i++)  
    	{
		    Vec4i L = leftLines[i];

		    leftSigXY += L[0] * L[1] + L[2] * L[3];
    		leftSigX += L[0] + L[2];
    		leftSigY += L[1] + L[3];
    		leftSigXX += L[0] * L[0] + L[2] * L[2];
		}
    	
    	// 왼쪽 라인 그리기
    	if (leftLines.size() != 0)
		{
			lefta1 = leftn * leftSigXY - leftSigX * leftSigY / leftn * leftSigXX - leftSigX * leftSigX;
			lefta0 = (leftSigY - leftSigX * lefta1) / leftn;
    		
    		double lX1 = (0 - lefta0) / lefta1;
    		double lX2 = (660 - lefta0) / lefta1;

    		line(imgL, Point(lX1, 0), Point(lX2, 660), Scalar(0, 0, 255), 3, LINE_AA);
    	}
    	
		// 오른쪽 시그마값 구하기
		for (int i = 0; i < rightLines.size(); i++)
    	{
		    Vec4i R = rightLines[i];

		    rightSigXY += R[0] * R[1] + R[2] * R[3];
    		rightSigX += R[0] + R[2];
    		rightSigY += R[1] + R[3];
    		rightSigXX += R[0] * R[0] + R[2] * R[2];
    	}
    	
    	// 오른쪽 라인 그리기
    	if (rightLines.size() != 0)
		{
    		righta1 = rightn * rightSigXY - rightSigX * rightSigY / rightn * rightSigXX - rightSigX * rightSigX;
			righta0 = (rightSigY - rightSigX * righta1) / rightn;
    		
    		double rX1 = (0 - righta0) / righta1;
    		double rX2 = (660 - righta0) / righta1;

    		line(imgR, Point(rX1, 0), Point(rX2, 660), Scalar(255, 0, 0), 3, LINE_AA);
    	}
    	
    	// 이미지 합치기
    	Mat houghResult;
    	hconcat(imgL, imgR, houghResult);


		// 출력
		imshow("원본 영상", src);
		imshow("HLS 영상", hls_img);
		imshow("Combine W and Y", imgResult);
		imshow("canny edge 후", cannyImg);
		imshow("hough line transform 후", houghResult);

   	    if (waitKey(40) == 0);
	}   
}


