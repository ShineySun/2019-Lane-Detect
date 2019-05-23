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

void transform(Point2f* src_vertices, Point2f* dst_vertices, Mat& src, Mat &dst){
    Mat M = getPerspectiveTransform(src_vertices, dst_vertices);
    warpPerspective(src, dst, M, dst.size(), INTER_LINEAR, BORDER_CONSTANT);
}

Mat remove_noise(Mat src)
{
  Mat output;

  GaussianBlur(src, output, cv::Size(3, 3), 0, 0);

  return output;
}

void enhancer(Mat &src, Mat &dst)
{
  dst = src.clone();
  dst = Scalar(0);

  int width = src.cols;
  int height = src.rows;

  double max,min;

  minMaxLoc(src, &min, &max);

  dst = (src - min) * (255.0 / (max-min));

  cout << max << ' ' << min << endl;
}


int main()
{
  VideoCapture video("project_video.mp4");

  if(!video.isOpened())
    printf("Failed to open the video\n");

  Mat src,dst;
  Mat gray_img,lab_img,luv_img,hsv_img,hsl_img;
  Mat gaussian_img;
  Mat binary_img;

  Mat left,right;

  int quit;

  vector<Mat> channel(12);

  vector<Mat> R_channel(12);
  vector<Mat> L_channel(12);

  vector<Mat> new_R_channel(12);
  vector<Mat> new_L_channel(12);

  Mat result_img;

  // Mat new_lab[3];
  // Mat new_luv[3];
  // Mat new_hsv[3];
  // Mat new_hsl[3];

  vector<Mat> new_lab(3);
  vector<Mat> new_luv(3);
  vector<Mat> new_hsv(3);
  vector<Mat> new_hsl(3);


  Point2f src_vertices[4];

  video >> src;

  src_vertices[0] = Point(src.cols/2-150,500); // 좌상
  src_vertices[1] = Point(src.cols/2+150,500); // 우상
  src_vertices[2] = Point(src.cols/2+600,700); // 우하
  src_vertices[3] = Point(src.cols/2-500,700); // 좌하

  Point2f dst_vertices[4];

  dst_vertices[0] = Point(0, 0);
  dst_vertices[1] = Point(480, 0);
  dst_vertices[2] = Point(480, 620);
  dst_vertices[3] = Point(0, 620);


  while(1)
  {
    video >> src;

    // Bird eyes view Transform
    Mat M = getPerspectiveTransform(src_vertices, dst_vertices);
    Mat dst(480, 480, CV_8UC3);

    warpPerspective(src, dst, M, dst.size(), INTER_LINEAR, BORDER_CONSTANT);

    // GaussianBlur
    gaussian_img = remove_noise(dst);

    // Set ROI
    Rect left_rect(0, 0, dst.cols/2, dst.rows);
    Rect right_rect(dst.cols/2, 0, dst.cols/2, dst.rows);

    left = dst(left_rect);
    right = dst(right_rect);

    // RGB TO LAB & RGB TO LUV & RGB TO HSV & RGB TO HLS
    cvtColor(gaussian_img,lab_img,CV_BGR2Lab);
    cvtColor(gaussian_img,luv_img,CV_BGR2Luv);
    cvtColor(gaussian_img,hsv_img,CV_BGR2HSV);
    cvtColor(gaussian_img,hsl_img,CV_BGR2HLS);

    // CREATE CHANNEL VECTOR
    vector<Mat> lab_plane;
    vector<Mat> luv_plane;
    vector<Mat> hsv_plane;
    vector<Mat> hsl_plane;

    // SPLIT CHANNEL
    split(lab_img,lab_plane);
    split(luv_img,luv_plane);
    split(hsv_img,hsv_plane);
    split(hsl_img,hsl_plane);

    // COLLECT ALL CHANNEL
    for(int i = 0; i<3; i++)
    {
      channel[i] = lab_plane[i];
      channel[i+3] = luv_plane[i];
      channel[i+6] = hsv_plane[i];
      channel[i+9] = hsl_plane[i];
    }

    // SET ROI
    for(int i = 0; i<12; i++)
    {
      R_channel[i] = channel[i](right_rect);
      L_channel[i] = channel[i](left_rect);
    }

    // ENHANCING IMAGE
    for(int i = 0; i < 12; i++)
    {
      enhancer(R_channel[i],new_R_channel[i]);
      enhancer(L_channel[i],new_L_channel[i]);
    }

    // THRESH HOLDING
    for(int i = 0; i<12; i++)
    {
      threshold(new_R_channel[i],new_R_channel[i],50,255,THRESH_BINARY);
      threshold(new_L_channel[i],new_L_channel[i],125,255,THRESH_BINARY);
    }

    // imshow("R_0",new_R_channel[0]);
    // imshow("R_1",new_R_channel[1]);
    // imshow("R_2",new_R_channel[2]);
    // imshow("R_3",new_R_channel[3]);
    // imshow("R_4",new_R_channel[4]);
    // imshow("R_5",new_R_channel[5]);
    // imshow("R_6",new_R_channel[6]);
    // imshow("R_7",new_R_channel[7]);
    // imshow("R_8",new_R_channel[8]);
    // imshow("R_9",new_R_channel[9]);
    // imshow("R_10",new_R_channel[10]);
    // imshow("R_11",new_R_channel[11]);   // Good
    imshow("DST",dst);
    imshow("SRC",src);

    imshow("L_0",new_L_channel[0]);
    imshow("L_1",new_L_channel[1]);
    imshow("L_2",new_L_channel[2]);
    imshow("L_3",new_L_channel[3]);
    imshow("L_4",new_L_channel[4]);
    imshow("L_5",new_L_channel[5]);
    imshow("L_6",new_L_channel[6]);
    imshow("L_7",new_L_channel[7]);
    imshow("L_8",new_L_channel[8]);
    imshow("L_9",new_L_channel[9]);
    imshow("L_10",new_L_channel[10]);
    imshow("L_11",new_L_channel[11]);


    quit = waitKey(33);

    if(quit == 27)
    {
      waitKey(10000);
      continue;
    }
  }
}
