#include <vector>
#include "opencv2/opencv.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include <iostream>
#include <algorithm>

using namespace std;
using namespace cv;

#define VIDEO           "project_video.mp4"
#define WIDTH           480
#define HEIGHT          700

bool cmp(const Vec2i &a, const Vec2i &b);  //  AverageOrigin을 위한 sort함수

class LANEDETECTOR
{
private:
  VideoCapture cap; //  동영상
  Mat src, dst, hsv, gray, gaussian, canny; //  기본 블러 씌울 것들
  Mat left, right, retLeft, retRight; //  결과를 ret에 저장
  Point2f src_vertices[4], dst_vertices[4]; //  src(기본 영상 내의 좌표), dst(버드뷰의 좌표)
  Rect rectLeft, rectRight; //  버드뷰의 왼쪽 프레임, 오른쪽 프레임
  vector<Vec4i> linesLeft, linesRight;  //  허프라인 한 것들
  int lre, lre2, rre, rre2; //  전프레임 이용하기 위한 멤버변수
public:
  LANEDETECTOR () : lre(1), lre2(1), rre(1), rre2(1) { initCapture(); initFrame(); initBirdViewFrame(); }
  void initCapture(); //  비디오 넣기
  void initFrame(); //  기본 틀 잡기
  void initBirdViewFrame(); //  버드뷰 틀
  void transform(Point2f* src_vertices_, Point2f* dst_vertices_, Mat& src_, Mat &dst_); //  버드뷰 변환
  Mat detectColor();  //  색 검출
  void coverBlur(Mat& result_); //  가우시안, 캐니 씌우기
  void Ransac(vector<Vec4i> lines); //  말만 랜섹이고 아직.. 구현은 못한...
  vector<Vec4i> Average(vector<Vec4i> lines, int height, int width, int step); //  hough lines P의 등분으로 나눈뒤 x좌표 와 y좌표 둘다 평균 하나씩
  vector<Vec4i> AverageOrigin(vector<Vec4i> lines, int height, int width, int step); //  hough lines P의 y좌표는 그대로 y좌표의 등분에 대하여 x좌표만 평균으로 바꾸기
  void linearRegression();  //  리니어 레그레이션
  Mat CenterRoi(Mat origin); //  ROi의 센터 정지선 및 여러가지를 위한 ROI 설정하기
  void operate(); //  종합 연산
};

void LANEDETECTOR::initCapture()
{
  VideoCapture tpcap(VIDEO);
  if(!tpcap.isOpened()){ cerr << "비어있엉 돌아가~!\n";}
  this->cap = tpcap;
  tpcap >> src;
}

void LANEDETECTOR::initFrame()
{
  Mat tpdst(HEIGHT, WIDTH, CV_8UC3);
  dst = tpdst;
  Rect rLeft(0, 0, WIDTH / 2, HEIGHT), rRight(WIDTH / 2, 0, WIDTH / 2, HEIGHT);
  rectLeft = rLeft;
  rectRight = rRight;
}

void LANEDETECTOR::initBirdViewFrame()
{   
  src_vertices[0] = Point(src.cols/2-150, 490);
  src_vertices[1] = Point(src.cols/2+190, 490);
  src_vertices[2] = Point(src.cols/2+600, 670);
  src_vertices[3] = Point(src.cols/2-500, 670);
  
  dst_vertices[0] = Point(0, 0);
  dst_vertices[1] = Point(WIDTH, 0);
  dst_vertices[2] = Point(WIDTH, HEIGHT);
  dst_vertices[3] = Point(0, HEIGHT);
}

void LANEDETECTOR::transform(Point2f* src_vertices_, Point2f* dst_vertices_, Mat& src_, Mat &dst_)
{
  Mat M = getPerspectiveTransform(src_vertices_, dst_vertices_);
  warpPerspective(src_, dst_, M, dst_.size(), INTER_LINEAR, BORDER_CONSTANT);
}

Mat LANEDETECTOR::detectColor()
{      
  Mat tpresult(HEIGHT, WIDTH, CV_8UC1, Scalar(0));
  cvtColor(dst, hsv, COLOR_BGR2HSV);
  cvtColor(dst, gray, COLOR_BGR2GRAY);
  left = hsv(rectLeft);
  right = hsv(rectRight);
  ///////////////////////////////////////////////////////////////////////////
  retLeft = hsv(rectLeft);
  retRight = hsv(rectRight);
  Mat tpLeft = hsv(rectLeft);
  Mat tpRight = hsv(rectRight);
  inRange(left, Scalar(19, 50, 50), Scalar(36, 255, 255), retLeft);   //왼쪽 노란색 검출
  inRange(left, Scalar(0, 0, 225), Scalar(180, 30, 255), tpLeft); //왼쪽 흰색 검출
  bitwise_or(tpLeft, retLeft, retLeft);
  inRange(left, Scalar(0, 180, 55), Scalar(20, 255, 200), tpLeft); //  파란색 검출
  bitwise_or(tpLeft, retLeft, retLeft);
  
  inRange(right, Scalar(19, 50, 50), Scalar(36, 255, 255), retRight);   //오른쪽 노란색 검출
  inRange(right, Scalar(0, 0, 225), Scalar(180, 30, 255), tpRight); //오른쪽 흰색 검출
  bitwise_or(tpRight, retRight, retRight);
  inRange(right, Scalar(0, 180, 55), Scalar(20, 255, 200), tpRight); //  파란색 검출
  bitwise_or(tpRight, retRight, retRight);
  ///////////////////////////////////////////////////////////////////////////
  inRange(left, Scalar(19,50,50),Scalar(36,255,255), left);   //왼쪽 노란색 검출
  inRange(right, Scalar(0,0,225), Scalar(180,30,255), right); //오른쪽 흰색 검출
  
  for(int y = 0; y < HEIGHT; y++)
  {
    for(int x = 0; x < (WIDTH / 2); x++){
      if(retLeft.at<uchar>(y,x) == 255)
      {
        tpresult.at<uchar>(y,x) = gray.at<uchar>(y,x);
      }
    }
    for(int x = (WIDTH / 2); x < WIDTH; x++){
      if(retRight.at<uchar>(y,x - (WIDTH / 2)) == 255)
      {
        tpresult.at<uchar>(y,x) = gray.at<uchar>(y,x);
      }
    }
  }
  return tpresult;
}

void LANEDETECTOR::coverBlur(Mat& result_)
{
  GaussianBlur(result_, gaussian, Size(7, 7), 0);
  Canny(gaussian, canny, 40, 90, 3);
  left = canny(rectLeft);
  right = canny(rectRight);
}

void LANEDETECTOR::Ransac(vector<Vec4i> lines)
{
  
}

vector<Vec4i> LANEDETECTOR::Average(vector<Vec4i> lines, int height, int width, int step) //  등분으로 나눈뒤 x좌표 와 y좌표 둘다 평균 하나씩
{
  vector<Vec4i> result;
  vector<Vec2i> tp;
  bool flag;
  for(int i = 0; i < lines.size(); i++)
  {
    Vec4i tpp = lines[i];
    Vec2i tp1, tp2;
    tp1[0] = tpp[0];
    tp1[1] = tpp[1];
    tp2[0] = tpp[2];
    tp2[1] = tpp[3];
    tp.push_back(tp1);
    tp.push_back(tp2);
  }
  //sort(tp.begin(), tp.end(), cmp);
  vector<Vec2i> resultTemp;
  int divideSize = height / step;

  for(int i = 0; i < step; i++)
  {
    double totalX = 0;
    double totalY = 0;
    double cnt = 0;
    for(int j = 0; j < tp.size() ; j++)
    {
      if((i * divideSize) <= tp[j][1] && (tp[j][1] < (i * divideSize) + divideSize))
      {
        totalX += tp[j][0];
        totalY += tp[j][1];
        cnt++;
      }
    }
    if(cnt != 0)
    {
      totalX /= (double)cnt;
      totalY /= (double)cnt;
      Vec2i temp;
      temp[0] = totalX;
      temp[1] = totalY;
      resultTemp.push_back(temp);
      totalX = 0;
      totalY = 0;
      cnt = 0;
    }
  }
  for(int i = 0; i < (resultTemp.size() / 2); i++)
  {
    Vec4i temp;
    temp[0] = resultTemp[i * 2][0];
    temp[1] = resultTemp[i * 2][1];
    temp[2] = resultTemp[(i * 2) + 1][0];
    temp[3] = resultTemp[(i * 2) + 1][1];
    result.push_back(temp);
  }
  return result;
}

vector<Vec4i> LANEDETECTOR::AverageOrigin(vector<Vec4i> lines, int height, int width, int step) //  y좌표는 그대로 x좌표만 평균으로 바꾸기
{
  vector<Vec4i> result;
  vector<Vec2i> tp;
  for(int i = 0; i < lines.size(); i++)
  {
    Vec4i tpp = lines[i];
    Vec2i tp1, tp2;
    tp1[0] = tpp[0];
    tp1[1] = tpp[1];
    tp2[0] = tpp[2];
    tp2[1] = tpp[3];
    tp.push_back(tp1);
    tp.push_back(tp2);
  }
  sort(tp.begin(), tp.end(), cmp);
  vector<Vec2i> resultTemp;
  int divideSize = height / step;

  for(int i = 0; i < step; i++)
  {
    double totalX = 0;
    double cnt = 0;
    for(int j = 0; j < tp.size() ; j++)
    {
      if((i * divideSize) <= tp[j][1] && (tp[j][1] < (i * divideSize) + divideSize))
      {
        totalX += tp[j][0];
        cnt++;
      }
    }
    if(cnt != 0)
    {
      totalX /= (double)cnt;
      Vec2i temp;
      temp[0] = totalX;
      temp[1] = i;
      resultTemp.push_back(temp);
      totalX = 0;
      cnt = 0;
    }
    else
    {
      Vec2i temp;
      temp[0] = totalX;
      temp[1] = i;
      resultTemp.push_back(temp);
    }
  }
  for(int i = 0; i < tp.size(); i++)
  {
    for(int j = 0; j < resultTemp.size(); j++)
    {
      if((j * divideSize) <= tp[i][1] && tp[i][1] < ((j * divideSize) + divideSize))
      {
        tp[i][0] = resultTemp[j][0];
        break;
      }
    }
  }
  for(int i = 0; i < (tp.size() / 2); i++)
  {
    Vec4i temp;
    temp[0] = tp[i * 2][0];
    temp[1] = tp[i * 2][1];
    temp[2] = tp[(i * 2) + 1][0];
    temp[3] = tp[(i * 2) + 1][1];
    result.push_back(temp);
  }
  return result;
}

void LANEDETECTOR::linearRegression()
{
  HoughLinesP(left, linesLeft, 1, CV_PI/180.0, 0, 0, 0);
  HoughLinesP(right, linesRight, 1, CV_PI/180.0, 0, 0, 0);
  // HoughLinesP(left, linesLeft, 1, CV_PI/180.0, 40, 90, 3);
  // HoughLinesP(right, linesRight, 1, CV_PI/180.0, 150, 40, 10);
  linesLeft = Average(linesLeft, HEIGHT, WIDTH, 140);
  linesRight = Average(linesRight, HEIGHT, WIDTH, 140);
  if(linesLeft.size() != 0)
  {
    double lsx = 0, lsy = 0, lnsxy = 0, lsxsy = 0, lnsx2 = 0, ls2x = 0, a0 = 0, a1 = 0, top = 0, bottom = 0;
    for(int i = 0; i < linesLeft.size(); i++)
    {
      Vec4i L = linesLeft[i];
      lnsxy += ((L[0] * L[1]) + (L[2] * L[3]));
      lsx += L[0] + L[2];
      lsy += L[1] + L[3];
      lnsx2 += (L[0] * L[0]) + (L[2] * L[2]);
    }
    lnsxy *= (linesLeft.size() * 2);
    lsxsy = lsx * lsy;
    lnsx2 *= (linesLeft.size() * 2);
    ls2x = lsx * lsx;
    top = lnsxy - lsxsy;
    bottom = lnsx2 - ls2x;
    if(bottom != 0 && top != 0)
    {
      a1 = top / bottom;
      //if(a1 < -10 || a1 > 10){  //  기울기 값으로 튀는거 잡기
      lsx /= (linesLeft.size() * 2);
      lsy /= (linesLeft.size() * 2);
      a0 = lsy - (lsx * a1);
      lre = ((-1) * a0) / a1;
      lre2 = (HEIGHT - a0) / a1;
      //}
    }
  }
  if(linesRight.size() != 0)
  {
    double lsx = 0, lsy = 0, lnsxy = 0, lsxsy = 0, lnsx2 = 0, ls2x = 0, a0 = 0, a1 = 0, top = 0, bottom = 0;
    for(int i = 0; i < linesRight.size(); i++)
    {
      Vec4i L = linesRight[i];
      lnsxy += ((L[0] * L[1]) + (L[2] * L[3]));
      lsx += L[0] + L[2];
      lsy += L[1] + L[3];
      lnsx2 += (L[0] * L[0]) + (L[2] * L[2]);
    }
    lnsxy *= (linesRight.size() * 2);
    lsxsy = lsx * lsy;
    lnsx2 *= (linesRight.size() * 2);
    ls2x = lsx * lsx;
    top = lnsxy - lsxsy;
    bottom = lnsx2 - ls2x;
    if(bottom != 0 && top != 0)
    {
      a1 = top / bottom;
      //if(a1 < -10 || a1 > 10){
      lsx /= (linesRight.size() * 2);
      lsy /= (linesRight.size() * 2);
      a0 = lsy - (lsx * a1);
      rre = ((-1) * a0) / a1;
      rre2 = (HEIGHT - a0) / a1;
      //}
    }
  }
  //라인 긋기
  left = dst(rectLeft);
  right = dst(rectRight);
  line(left, Point(lre,0), Point(lre2, HEIGHT), Scalar(0, 0, 255), 4, LINE_AA );
  line(right, Point(rre,0), Point(rre2 , HEIGHT), Scalar(255, 0, 0), 4, LINE_AA );
  for(int y = 0; y < HEIGHT; y++)
  {
    for(int x = 0; x < WIDTH; x++){
      if(left.at<Vec3b>(y,x)[2] != dst.at<Vec3b>(y,x)[2])
      {
        dst.at<Vec3b>(y,x)[2] = 255;
      }
    }
    for(int x = (WIDTH / 2); x < WIDTH; x++){
      if(right.at<Vec3b>(y,x - (WIDTH / 2))[2] != dst.at<Vec3b>(y,x)[2])
      {
        dst.at<Vec3b>(y,x)[2] = 255;
      }
    }
  }
}

Mat LANEDETECTOR::CenterRoi(Mat origin)
{
  Mat result;
  int distanceOfLane = 15;
  vector<Point> center_vertices;
  
  center_vertices.push_back(Point(lre + distanceOfLane, 0));
  center_vertices.push_back(Point(rre + (WIDTH / 2) - distanceOfLane, 0));
  center_vertices.push_back(Point(rre2 + (WIDTH / 2) - distanceOfLane, HEIGHT));
  center_vertices.push_back(Point(lre2 + distanceOfLane, HEIGHT));

  Mat roi(HEIGHT, WIDTH, CV_8UC3, Scalar(0));

  fillConvexPoly(roi, center_vertices, Scalar(255, 255, 255));
  Mat filteredImg;
  origin.copyTo(filteredImg, roi);
  return filteredImg;
}

void LANEDETECTOR::operate()
{
  while(1)
  {   
    Mat result(HEIGHT, WIDTH, CV_8UC1);
    cap>>src;
    if(src.empty()){ cerr<<"비어있엉 돌아가...!!\n"; break; }
    imshow("src", src);
    LANEDETECTOR::transform(src_vertices, dst_vertices, src, dst);  //  버드뷰
    result = detectColor();
    coverBlur(result);  //  result를 gaussian 및 canny를 씌워서 dst에 저장
    linearRegression(); //  dst를 가지고 houghlinesp를 사용 후, 
    Mat tp = CenterRoi(dst);
    imshow("tp", tp);
    imshow("result", dst);
    if (waitKey(10) == 27) // waitkey는 시간별로 프레임 끊는 것이고, 27이 ESC버튼이라고 생각 사용자의 입력을 받아 뭐라는뎅 이건 이해가안됨
		break;
  }
}

bool cmp(const Vec2i &a, const Vec2i &b)  //  AverageOrigin을 위한 sort함수
{
  if(a[1] < b[1])
    return true;
  else
    return false;
}

int main()
{
  LANEDETECTOR lane;
  lane.operate();
  return 0;
}