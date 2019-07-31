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

#include <gsl/gsl_fit.h>

using namespace std;
using namespace cv;



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

  //threshold(hsv_plane[1],s_thresh,125,255,CV_THRESH_BINARY);
  //imshow("s_thresh",s_thresh);

  // WHITE MASK
  inRange(hls_img, Scalar(0, 200, 0), Scalar(255, 255, 255), wImgMask);
  // YELLOW MASK
  inRange(hls_img, Scalar(15, 30, 115), Scalar(35, 205, 255), yImgMask);

  imgMask = wImgMask | yImgMask;

  Mat result;
  bitwise_and(src,src,result,imgMask);

  return result;
}

void make_histogram(Mat &src, vector<int> &counter,Mat &dst)
{
  int total_counter  = 0;
  // cout << "가로 : " << src.cols << " 세로 : " << src.rows << endl;
  // cout << "vector size : " << counter.size() << endl;
  int x_counter = 0;

    for(int x = 0; x < src.cols; x++)
    {
      // 흰색 픽셀 개수 저장
      for(int y = 0; y < src.rows; y++)
      {
        if(src.at<uchar>(y,x) == 255)
        {
          counter[x]++;
          total_counter++;
          // circle(dst,Point(x,y),5,Scalar(0,255,0),5);
          // waitKey(1000);
        }
      }
      x_counter++;
      // cout << x << " 번 : " << counter[x] << "개" << endl;
    }
    // cout << x_counter << endl;


    // cout << "total" << total_counter << endl;
    int max = *max_element(counter.begin(), counter.end());
    int min = *min_element(counter.begin(), counter.end());

    // cout << "Max : " << max << " Min : " << min << endl;

    int hist_w = 1280;
    int hist_h = 300;
    int bin_w = 1;

    // cout << "bin_w : " << bin_w << endl;

    Mat histogram = Mat(hist_h,hist_w, CV_8UC1,Scalar(0,0,0));

    for(int i = 1; i < 1281; i++)
    {
      line(histogram, Point(bin_w*(i-1),hist_h - cvRound(counter[i-1]*10000/total_counter)),Point(bin_w*i, hist_h - cvRound(counter[i]*10000/total_counter)),Scalar(255,255,255),2,8,0);
      // line(histogram, Point(bin_w*640,0), Point(bin_w*640,720),Scalar(255,0,0),2,8,0);
      // cout << " 좌표값 : " << bin_w*(i-1) << ", " << hist_h - cvRound(counter[i-1]*1000/total_counter) << ' ';
    }
    // cout << endl;

    imshow("histogram",histogram);


    // histogram.setTo(Scalar(0));
    //
    //
    // for(int y = 0; y < src.rows; y++)
    // {
    //   circle(histogram,Point(y,counter[y]),5,Scalar(255,255,255),5);
    // }
    //
    // imshow("hist",histogram);
}

void set_roi(Mat &src, Mat &left_img, Mat &right_img)
{
  Mat temp = src.clone();

  //cout << topView_img.rows/2 << ' ' << topView_img.cols << endl;
  //
  Rect left_rect(0, 0, temp.cols/2, temp.rows);
  Rect right_rect(temp.cols/2, 0, temp.cols/2, temp.rows);
  //
  // imshow("Temp",temp);
  //left_img = topView_img.clone();
  //right_img = topView_img.clone();
  left_img = temp(left_rect);
  right_img = temp(right_rect);
}

void canny(Mat &left_bi, Mat &right_bi)
{
  Canny(left_bi,left_bi,150,270,5);
  Canny(right_bi,right_bi,150,270,5);
}

bool cmp(const Vec2i &a, const Vec2i &b)
{
  return a[1] < b[1];
}

void detect_left_line(Mat &left_img,vector<Vec4i> &left_lines, Mat &dst, Mat &imgResult, int distance)
{
  HoughLinesP(left_img, left_lines, 1, CV_PI/180, 0, 0, 0 );

  if(!left_lines.size() && distance == 0)
  {
    cout << "No Left Line" << endl;
    return;
  }

  if(!left_lines.size() && distance != 0)
  {
    cout << "No Right Line" << endl;
    return;
  }

  int vec_2i_size = left_lines.size()*2;

  vector<Vec2i> lines(vec_2i_size);

  // cout << lines.size() << endl;
  // cout << vec_2i_size << endl;
  int lines_idx = 0;

  for(int i = 0; i < left_lines.size(); i++)
  {
    Vec4i line = left_lines[i];

    int x1 = line[0];
    int y1 = line[1];
    int x2 = line[2];
    int y2 = line[3];

    lines[lines_idx][0] = x1;
    lines[lines_idx++][1] = y1;

    lines[lines_idx][0] = x2;
    lines[lines_idx++][1] = y2;
  }

  sort(lines.begin(), lines.end(), cmp);

  double real_lines_x[10000];
  double real_lines_y[10000];

  int step = 60;
  int height = dst.rows;



  int divide_size = height/step;
  int real_line_idx = 0;

  for(int i = 0; i < step; i++)
  {
    double sum_x = 0;
    double sum_y = 0;
    double cnt = 0 ;

    for(int j = 0; j < lines.size(); j++)
    {
      if(((i*divide_size) <= lines[j][1]) && ((lines[j][1]) < ((i+1)*divide_size)))
      {
        sum_x += lines[j][0];
        sum_y += lines[j][1];
        cnt++;
      }
    }

    if(cnt)
    {
      sum_x /= (double)cnt;
      sum_y /= (double)cnt;

      real_lines_x[real_line_idx] = sum_x;
      real_lines_y[real_line_idx++] = sum_y;

      sum_x = 0;
      sum_y = 0;
      cnt = 0;
    }

  }

  // for(int i = 0; i < real_line_idx; i++)
  // {
  //   circle(imgResult,Point(real_lines_x[i],real_lines_y[i]),6,Scalar(0,0,255),CV_FILLED);
  // }

  // RANSAC

  srand(time(NULL));

  double noise_sigma = 100;

  Mat A(real_line_idx,2,CV_64FC1);
  Mat B(real_line_idx,1,CV_64FC1);

  for( int i=0 ; i < real_line_idx ; i++ )
	{
		A.at<double>(i,0) = real_lines_x[i] ;
	}

	for( int i=0 ; i < real_line_idx ; i++ )
	{
		A.at<double>(i,1) = 1.0 ;
	}

	for( int i=0 ; i < real_line_idx ; i++ )
	{
		B.at<double>(i,0) = real_lines_y[i] ;
	}

  int n_data = real_line_idx;
  int N = real_line_idx;
  double T = noise_sigma;

  int n_sample = 3;
  int max_cnt = 0;

  Mat best_model(2,1,CV_64FC1);

  for(int i = 0; i < N; i++)
  {
    // random sampling - 3 Point
    int k[3] = {-1, };
    k[0] = floor((rand()%100+1))+1;

    do
		{
			k[1] = floor((rand()%100+1))+1;
		}while(k[1]==k[0] || k[1]<0) ;

    do
		{
			k[2] = floor((rand()%100+1))+1;
		}while(k[2]==k[0] || k[2]==k[1] || k[2]<0);

    cv::Mat AA(3,2,CV_64FC1) ;
		cv::Mat BB(3,1, CV_64FC1) ;

    for( int j=0 ; j<3 ; j++ )
		{
			// AA.at<double>(j,0) = real_left_lines_x[k[j]] * real_left_lines_x[k[j]] ;
			AA.at<double>(j,0) = real_lines_x[k[j]] ;
			AA.at<double>(j,1) = 1.0 ;

			BB.at<double>(j,0) = real_lines_y[k[j]] ;
		}

    cv::Mat AA_pinv(3,2,CV_64FC1) ;
		invert(AA, AA_pinv, cv::DECOMP_SVD);

    cv::Mat X = AA_pinv * BB ;

    cv::Mat residual(real_line_idx,1,CV_64FC1) ;
		residual = cv::abs(B-A*X) ;

    int ransac_cnt = 0;

    for( int j=0 ; j<real_line_idx ; j++ )
		{
			double data = residual.at<double>(j,0) ;

			if( data < T )
			{
				ransac_cnt++ ;
			}
		}

    if( ransac_cnt > max_cnt )
		{
			best_model = X ;
			max_cnt = ransac_cnt ;
		}

  }

  cv::Mat residual = cv::abs(A*best_model - B) ;
	std::vector<int> vec_index ;

  for( int i=0 ; i<real_line_idx ; i++ )
	{
		double data = residual.at<double>(i, 0) ;

    if( data < T )
		{
      // cout << "vector push_back" << endl;
			vec_index.push_back(i) ;
		}

	}

  cv::Mat A2(vec_index.size(),2, CV_64FC1) ;
	cv::Mat B2(vec_index.size(),1, CV_64FC1) ;

  for( int i=0 ; i<vec_index.size() ; i++ )
	{
		// A2.at<double>(i,0) = real_left_lines_x[vec_index[i]] * real_left_lines_x[vec_index[i]]  ;
		A2.at<double>(i,0) = real_lines_x[vec_index[i]] ;
		A2.at<double>(i,1) = 1.0 ;

		B2.at<double>(i,0) = real_lines_y[vec_index[i]] ;
	}

  if(!vec_index.size())
  {
    cout << "NO LINE" << endl;
    return;
  }
  cout << vec_index.size() << endl;

  cv::Mat A2_pinv(2,vec_index.size(),CV_64FC1) ;
	invert(A2, A2_pinv, cv::DECOMP_SVD);

  cv::Mat X = A2_pinv * B2 ;

  cv::Mat F = A*X ;

  int interval = 1;

  for( int iy=0 ; iy<real_line_idx ; iy++ )
	{
		cv::circle(imgResult, cv::Point(real_lines_x[iy]*interval + distance, real_lines_y[iy]*interval) ,3, cv::Scalar(0,0,255), CV_FILLED) ;

		double data = F.at<double>(iy,0) ;

		cv::circle(imgResult, cv::Point(real_lines_x[iy]*interval + distance, data*interval) ,5, cv::Scalar(0,255,0), CV_FILLED) ;
	}

  imshow("img_result",imgResult);
}

int main()
{
  // VideoCapture video("project_video.mp4");
  VideoCapture video(1);

  if(!video.isOpened())
    printf("Failed to open the video\n");

  Mat src(480,680,CV_8UC3);
  Mat dst(480,680,CV_8UC3);

  Mat hls_img;

  int quit;

  video >> src;

  cout << src.cols << src.rows << endl;
  // 680  480

  Point2f src_vertices[4];

  // src_vertices[0] = Point(src.cols/2 - 100,450);   // 좌상
  // src_vertices[1] = Point(src.cols/2 + 100, 450);  // 우상
  // src_vertices[2] = Point(src.cols/2 + 500, src.rows - 50); // 우하
  // src_vertices[3] = Point(src.cols/2 - 400, src.rows - 50);  // 좌하

  src_vertices[0] = Point(src.cols/2-140,300);   // 좌상
  src_vertices[1] = Point(src.cols/2+140, 300);  // 우상
  src_vertices[2] = Point(src.cols/2+200, src.rows); // 우하
  src_vertices[3] = Point(src.cols/2-200, src.rows);

  Point2f dst_vertices[4];

  dst_vertices[0] = Point(0, 0);
  dst_vertices[1] = Point(src.cols, 0);
  dst_vertices[2] = Point(src.cols, src.rows);
  dst_vertices[3] = Point(0, src.rows);

  while(1)
  {
      video >> src;

      Point2f center_upper = Point(src.cols/2,0);
      Point2f center_lower = Point(src.cols/2,src.rows);

      //circle(src,center_upper,5,Scalar(0,255,0),5);
      //circle(src,center_lower,5,Scalar(0,255,0),5);

      //line(src,center_upper,center_lower,Scalar(0,255,0),5);
      //void line
      //(InputOutputArray img, Point pt1, Point pt2, const Scalar& color, int thickness=1, int lineType=LINE_8, int shift=0 )
      //
      for(int i = 0; i < 4; i++)
      {
        circle(src,src_vertices[i],5,Scalar(255,0,0),5);
      }

      transform(src_vertices,dst_vertices,src,dst);

      Mat result = convert_hsl(dst);

      Mat result_bi;

      cvtColor(result,result_bi,COLOR_BGR2GRAY);

      GaussianBlur(result_bi, result_bi, Size(3, 3), 0);

      threshold(result_bi,result_bi,0,255,THRESH_BINARY | THRESH_OTSU);

      vector<int> counter(src.cols,0);
      // cout << "src.rows : " << src.rows << endl;

      // make_histogram(result_bi, counter,dst);

      Mat left_img,right_img;

      set_roi(result_bi, left_img, right_img);

      canny(left_img,right_img);

      vector<Vec4i> left_lines;
      vector<Vec4i> right_lines;

      cv::Mat imgResult(src.rows, src.cols,CV_8UC3) ;
    	imgResult = cv::Scalar(0);

      detect_left_line(left_img,left_lines, dst, imgResult,0);
      detect_left_line(right_img,right_lines,dst, imgResult, src.cols/2 );












      imshow("src",src);
      imshow("dst",dst);
      imshow("right_img",right_img);
      imshow("left_img", left_img);

      // imshow("h_channel",hls_plane[0]);
      // imshow("l_channel",hls_plane[1]);
      // imshow("s_channel",hls_plane[2]);
      // imshow("White",wImgMask);
      // imshow("Yellow",yImgMask);
      // imshow("Total",imgMask);
      // imshow("Result",result);
      // imshow("Result_bi", result_bi);

      quit = waitKey(33);

      if(quit == 27)
      {
        waitKey(10000);
        continue;
      }




  }

}
