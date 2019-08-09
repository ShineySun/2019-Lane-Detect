#include <iostream>
#include <vector>
#include <cstring>
#include <string>
#include <time.h>
#include <math.h>
#include <algorithm>
#include <cv.h>

#include "opencv2/opencv.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

enum MODE{
  GO_STRAIGHT,
  GO_RIGHT,
  GO_LEFT
};

MODE GO_MODE = GO_STRAIGHT;

const int WIDTH = 680;
const int HEIGHT = 480;

bool is_left = false;
bool is_right = false;

double slope_threshold = 90;

const Point default_left_vertices[4] =
{
  Point(0,0),
  Point(WIDTH/5, 0),
  Point(WIDTH/5,HEIGHT),
  Point(0,HEIGHT)
};

Point left_vertices[4] =
{
  Point(0,0),
  Point(WIDTH/5, 0),
  Point(WIDTH/5,HEIGHT),
  Point(0,HEIGHT)
};

const Point default_right_vertices[4] =
{
  Point(WIDTH/5*4,0),
  Point(WIDTH, 0),
  Point(WIDTH,HEIGHT),
  Point(WIDTH/5*4,HEIGHT)
};

Point right_vertices[4] =
{
  Point(WIDTH/5*4,0),
  Point(WIDTH, 0),
  Point(WIDTH,HEIGHT),
  Point(WIDTH/5*4,HEIGHT)
};

void set_left_roi(Mat& img, Mat& img_ROI){

  vector <Point> Left_Point;

  for(int i = 0; i < 4; i++)
  {
    Left_Point.push_back(default_left_vertices[i]);
  }

  Mat roi(img.rows, img.cols, CV_8U, Scalar(0));

  fillConvexPoly(roi, Left_Point, Scalar(255));

  Mat filteredImg_Left;
  img.copyTo(filteredImg_Left, roi);

  img_ROI = filteredImg_Left.clone();

}

void set_right_roi(Mat& img, Mat& img_ROI){

  vector <Point> Right_Point;

  for(int i = 0; i < 4; i++)
  {
    Right_Point.push_back(default_right_vertices[i]);
  }

  Mat roi(img.rows, img.cols, CV_8U, Scalar(0));

  fillConvexPoly(roi, Right_Point, Scalar(255));

  Mat filteredImg_Right;

  img.copyTo(filteredImg_Right, roi);

  img_ROI = filteredImg_Right.clone();

}


void transform(Point2f* src_vertices, Point2f* dst_vertices, Mat& src, Mat &dst){
    Mat M = getPerspectiveTransform(src_vertices, dst_vertices);
    warpPerspective(src, dst, M, dst.size(), INTER_LINEAR, BORDER_CONSTANT);
}

Mat convert_hsl(Mat &src)
{
  Mat hls_img;
  //Mat hsv_img;

  cvtColor(src,hls_img,COLOR_BGR2HLS);
  //cvtColor(src,hsv_img,COLOR_BGR2HSV);

  // vector<Mat> hsv_plane;
  // split(hsv_img,hsv_plane);

  //imshow("s_channel",hsv_plane[1]);
  //Mat s_thresh;
  Mat wImgMask, yImgMask, imgMask;
  Mat R_wImgMask, g_ImgMask;
  //threshold(hsv_plane[1],s_thresh,125,255,CV_THRESH_BINARY);
  //imshow("s_thresh",s_thresh);

  //RGB WHITE MASK
  inRange(src,Scalar(100,100,200),Scalar(255,255,255),R_wImgMask);

  // WHITE MASK
  inRange(hls_img, Scalar(0, 150, 0), Scalar(255, 255, 255), wImgMask);
  // inRange(hls_img, Scalar(0, 0, 225), Scalar(180, 30, 255), wImgMask);
  // YELLOW MASK
  // inRange(hls_img, Scalar(15, 30, 30), Scalar(65, 105, 255), yImgMask);
  // inRange(hls_img, Scalar(15, 30, 115), Scalar(35, 204, 255), yImgMask);
  inRange(hls_img, Scalar(15, 100, 115), Scalar(35, 204, 255), yImgMask);
  // inRange(hls_img, Scalar(0, 0, 0), Scalar(255, 255, 255), yImgMask);

  // inRange(hls_img,Scalar(15, 15, 20),Scalar(140, 100, 80),g_ImgMask);


  // imshow("RGB White Mask",R_wImgMask);
  // imshow("White Mask", wImgMask);
  // imshow("Yellow Mask", yImgMask);
  // imshow("Green Mask", g_ImgMask);

  imgMask = wImgMask | yImgMask;

  Mat result;

  bitwise_and(src,src,result,imgMask);

  // cvtColor(result,result,COLOR_BGR2HLS);
  // imshow("Result",result);

  return result;
}

int maximum(int straight_num, int left_num,int right_num)
{
  int max = straight_num;

  if(max < left_num)
    max = left_num;
  if(max < right_num)
    max = right_num;

  if(max == straight_num) return 1;
  else if(max == left_num) return 2;
  else if(max == right_num) return 3;
}

bool cmp(const Vec2i &a, const Vec2i &b)
{
  return a[1] < b[1];
}


void data_preprocess(Mat &src,vector<Vec4i> lines, double *real_line_x, double *real_line_y, vector<Vec2i> &lines_vec2i)
{
  if(lines.size() < 10)
  {
    printf("No Line\n");
    return;
  }


  // Number of Sub Point
  int straight_num = 0;
  int right_num = 0;
  int left_num = 0;

  // Array Of GO_MODE
  double straight_x[10000];
  double straight_y[10000];

  double right_x[10000];
  double right_y[10000];

  double left_x[10000];
  double left_y[10000];

  for(int i = 0; i < lines.size(); i++)
  {
    Vec4i line = lines[i];

    int x1 = line[0];
    int y1 = line[1];
    int x2 = line[2];
    int y2 = line[3];


    double dx = x1 - x2;
    double dy = y1 - y2;
    double angle;

    if(dx == 0)
    {
      angle = 90;
    }
    else{

    double r = sqrt(dx*dx + dy*dy); // distance

    angle = atan2(dy,dx);
    angle = angle*180/3.14;
    }

    if(angle < 0)
    {
      angle += 180;
    }

    if(angle >= 80 && angle < 100)
    {
      straight_x[straight_num] = x1;
      straight_y[straight_num++] = y1;

      straight_x[straight_num] = x2;
      straight_y[straight_num++] = y2;

      circle(src,Point(x1,y1),5,Scalar(0,255,0),5);
      circle(src,Point(x2,y2),5,Scalar(0,255,0),5);
    }
    else if(angle <= 150 && angle > 100)
    {
      left_x[left_num] = x1;
      left_y[left_num++] = y1;

      left_x[left_num] = x2;
      left_y[left_num++] = y2;

      circle(src,Point(x1,y1),5,Scalar(255,0,0),5);
      circle(src,Point(x2,y2),5,Scalar(255,0,0),5);
    }
    else if(angle >= 0 && angle < 80)
    {
      right_x[right_num] = x1;
      right_y[right_num++] = y1;

      right_x[right_num] = x2;
      right_y[right_num++] = y2;

      circle(src,Point(x1,y1),5,Scalar(0,0,255),5);
      circle(src,Point(x2,y2),5,Scalar(0,0,255),5);
    }

  }

  int biggest = maximum(straight_num, left_num, right_num);

  if(biggest == 1)
  {
    for(int i = 0; i < straight_num; i++)
    {
      // lines_vec2i[i][0] = straight_x[i];
      // lines_vec2i[i][1] = straight_y[i];
      lines_vec2i.push_back(Vec2i(straight_x[i],straight_y[i]));
    }

    GO_MODE = GO_STRAIGHT;

    cout << "GO_STRAIGHT" << endl;

    // return straight_num;
  }

  else if(biggest == 2)
  {
    for(int i = 0; i < left_num; i++)
    {
      // lines_vec2i[i][0] = left_x[i];
      // lines_vec2i[i][1] = left_y[i];
      lines_vec2i.push_back(Vec2i(left_x[i],left_y[i]));
    }

    GO_MODE = GO_LEFT;

    cout << "GO_LEFT" << endl;

    // return left_num;
  }

  else if(biggest == 3)
  {
    for(int i = 0; i < right_num; i++)
    {
      // lines_vec2i[i][0] = right_x[i];
      // lines_vec2i[i][0] = right_y[i];
      lines_vec2i.push_back(Vec2i(right_x[i],right_y[i]));
    }

    GO_MODE = GO_RIGHT;

    cout << "GO_RIGHT" << endl;

    // return right_num;
  }

  sort(lines_vec2i.begin(), lines_vec2i.end(), cmp);

  int pre_y = lines_vec2i[0][1];
  int x_min = lines_vec2i[0][0];

  int real_line_idx = 0;
  int pre_y_count = 0;

  for(int i = 0; i < lines_vec2i.size(); i++)
  {
    if(pre_y == lines_vec2i[i][1])
    {
      pre_y_count++;

      if(x_min > lines_vec2i[i][0])
        x_min = lines_vec2i[i][0];
    }
    else if(pre_y != lines[i][1] && pre_y_count > 3)
    {
      real_line_x[real_line_idx] = x_min;
      real_line_y[real_line_idx++] = pre_y;

      circle(src,Point(x_min,pre_y),5,Scalar(255,255,0),5);
      //circle(src,Point(x1,y1),5,Scalar(0,0,255),5);
      pre_y = lines_vec2i[i][1];
      x_min = lines_vec2i[i][0];
      pre_y_count = 0;
    }
    else if(pre_y != lines[i][1] && pre_y_count <= 3)
    {
      pre_y = lines_vec2i[i][1];
      x_min = lines_vec2i[i][0];
      pre_y_count = 0;
    }
  }
}

void canny(Mat &left_bi, Mat &right_bi)
{
  Canny(left_bi,left_bi,150,270,5);
  Canny(right_bi,right_bi,150,270,5);
}

void houghlinesP(Mat &left_img, Mat &right_img, vector<Vec4i> &left_lines, vector<Vec4i> &right_lines)
{
  HoughLinesP(left_img, left_lines, 1, CV_PI/180, 0, 0, 0 );

  HoughLinesP(right_img, right_lines, 1, CV_PI/180, 5, 10, 10 );
}

// void lab_threshold(Mat lab_img)
// {
//   /*
//    Threshold the input image to the B-channel of the LAB color space.
//       Parameters:
//           img: LAB image.
//           thresh: Minimum and Maximum color intensity.
//   */
//   // Create Mat vector to store each plane(L,A,B)
//   vector<Mat> lab_plane;
//
//   // Split the channel
//   split(lab_img,lab_plane);
//
//   double min,max;
//
//   imshow("b_plane1",lab_plane[2]);
//
//   // Search min & max value
//   minMaxLoc(lab_plane[2],&min,&max);
//
//   if(max > 175)
//   {
//     lab_plane[2] = lab_plane[2]*(255/max);
//   }
//
//   threshold(lab_plane[2],lab_plane[2],205,255,THRESH_BINARY);
//
//   imshow("b_plane",lab_plane[2]);
//
// }

void color_threshold(Mat src)
{
  // Create Mat class
  Mat hls_img, lab_img;

  // Convert color space (rgb -> hsl , lab)
  cvtColor(src,hls_img,COLOR_BGR2HLS);
  cvtColor(src,lab_img,COLOR_BGR2Lab);

  // lab_threshold(lab_img);

}

int main(int argc, char**argv)
{
  // Capture the Video
  VideoCapture video("for_lane3.avi");

  // Dealing Exception
  if(!video.isOpened())
    printf("Failed to open the video\n");

  // Create Mat class
  // src (raw data), dst (bird view transformation)
  Mat src, dst;

  // Create integer value for using waitKey
  int quit;

  // VideoCapture >> Mat;
  video >> src;

  // Setting vertices for bird eyes view
  Point2f src_vertices[4];

  src_vertices[0] = Point(src.cols/2 - 200,350);      // upper left
  src_vertices[1] = Point(src.cols/2 + 200, 350);     // upper right
  src_vertices[2] = Point(src.cols, src.rows - 50);   // lower right
  src_vertices[3] = Point(0, src.rows - 50);          // lower left

  Point2f dst_vertices[4];

  dst_vertices[0] = Point(0, 0);                      // upper left
  dst_vertices[1] = Point(src.cols, 0);               // upper right
  dst_vertices[2] = Point(src.cols, src.rows);        // lower right
  dst_vertices[3] = Point(0, src.rows);               // lower left

  while(1)
  {
    // Capture to Mat class
    video >> src;

    // Bird eyes view transformation
    transform(src_vertices,dst_vertices,src,dst);

    // Color Thresholding
    Mat color_filtered_img = convert_hsl(dst);

    // Gray Scaling
    Mat gray_img;

    cvtColor(color_filtered_img,gray_img,COLOR_BGR2GRAY);

    // GaussianBlur
    GaussianBlur(gray_img,gray_img,Size(3,3),0);

    // Binarization
    Mat binary_img;

    threshold(gray_img, binary_img,0,255,THRESH_BINARY | THRESH_OTSU);

    // Dilated Img
    Mat dilated_img;

    dilate(binary_img,dilated_img,Mat());

    // Set ROI
    Mat left_img, right_img;

    if(!is_left && !is_right)
    {
      // Default ROI
      set_left_roi(dilated_img, left_img);
      set_right_roi(dilated_img, right_img);
    }
    else
    {
      // SETTING ROI

    }

    // Operate Canny algorithm
    canny(left_img, right_img);

    // Operate HoughLinesP
    vector<Vec4i> left_lines;
    vector<Vec4i> right_lines;

    houghlinesP(left_img, right_img, left_lines, right_lines);

    // Preprocessed data
    vector<Vec2i> left_lines_vec2i;
    vector<Vec2i> right_lines_vec2i;
    // array of real line
    double real_left_line_x[10000];
    double real_left_line_y[10000];
    int real_left_line_idx = 0;

    double real_right_line_x[10000];
    double real_right_line_y[10000];
    int real_right_line_idx = 0;

    // Slope Thresholding & Raw Data Preprocessing
    data_preprocess(dst,left_lines, real_left_line_x, real_left_line_y, left_lines_vec2i);
    // data_preprocess(right_lines);







    // Showing the Img
    imshow("src",src);
    imshow("dst",dst);
    imshow("color_filtered_img",color_filtered_img);
    imshow("binary_img",binary_img);
    imshow("Dilated_img",dilated_img);
    imshow("left_img",left_img);
    imshow("right_img",right_img);

    quit = waitKey(1);

    if(quit == 27)
    {
      waitKey(10000);
      continue;
    }

  }

}
