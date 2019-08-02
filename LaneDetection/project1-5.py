#색변환, roi, 허프라인 함수로 만들기
#부동소수점 오류 잡기

#8월부터 ros공부 시작이니까 미리미리 영상다시보기
#유튜브나 깃에서 자료찾아서 분석해서 공부해오기


import cv2
import numpy as np

def draw_lines(img, lines, color=[0, 0, 255], thickness=4): # 선 그리기
    sig_xy=0;
    sig_x=0;
    sig_y=0;
    sig_xx=0;
    count=0;

    for line in lines:
            for x1,y1,x2,y2 in line:
                sig_xy+=x1*y1+x2*y2;
                sig_x+=x1+x2;
                sig_y=y1+y2;
                sig_xx+=x1*x1+x2*x2;
            count+=1;
    a1=((count*2)*(sig_xy)-(sig_x*sig_y))/((count*2)*(sig_xx)-(sig_x)*(sig_x));
    a0=(sig_y/(count*2))-(sig_x/(count*2))*a1;

    #if(a1 is not None or len(a1)==0) :
    resultx1=int(-a0/a1);
    resultx2=int((620-a0)/a1);

    return cv2.line(img, (resultx1,0), (resultx2,620), color, thickness)


def hough_transform(img, rect, rho, theta, threshold, min_line_len, max_line_gap): # 허프 변환
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    #line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    #print(lines.size())

    if (lines is None or len(lines) == 0):
      return rect

    draw_lines(rect, lines)
    return rect

def colorch(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2HLS);

def region_of_interest(image,width,height1,height2,roiw,roih):
    src=np.array([[width/2-150,height1],[width/2+150,height1],[width/2+650,height2],[width/2-550,height2]],np.float32)
    dst=np.array([[0,0],[roiw,0],[roiw,roih],[0,roih]],np.float32)
    M=cv2.getPerspectiveTransform(src,dst);
    return cv2.warpPerspective(image,M,(roiw,roih))

cap=cv2.VideoCapture('project_video.mp4')

while(cap.isOpened()):
    #프레임 읽기(한 프레임씩)
    ret, image=cap.read()

    warp=region_of_interest(image,1280,500,680,480,620)

    #색뱐환
    hls_img=colorch(warp)

    #채녈 분리
    hls_plane=cv2.split(hls_img);

    #색검출
    lower_white=(0,200,0)
    higher_white=(255,255,255)
    img_mask1=cv2.inRange(hls_img, lower_white, higher_white)

    lower_yellow=(20,40,100)
    higher_yellow=(35,200,255)
    img_mask2=cv2.inRange(hls_img, lower_yellow, higher_yellow)

    img_mask=cv2.bitwise_or(img_mask1,img_mask2,img_mask1)

    img_result=cv2.bitwise_and(warp,warp,mask=img_mask)

    #각각 분리한 채녈을 이진화하기
    #ret,gray_result=cv2.threshold(gray_img,110,255,cv2.THRESH_BINARY);
    gray_img=cv2.cvtColor(img_result,cv2.COLOR_BGR2HLS);

    ret,hls_result=cv2.threshold(gray_img,50,255,cv2.THRESH_BINARY);

    cv2.GaussianBlur(hls_result,(5,5),0)
    canny_img = cv2.Canny(hls_result,10,230, apertureSize=3)

    #roi로 왼쪽 오른쪽 치산 이미지 반으로 나눠서 houghlines/houghlinesp 따로따로 하기
    #->for문 계속돌려서 점들을 더해서 평균내주기
    #line함수써서 점 두개 이어오기

    roiL=canny_img[0:620,0:240]
    roiR=canny_img[0:620,240:480]

    imgL=warp[0:620,0:240]
    imgR=warp[0:620,240:480]

    #변수 값 바꿔보기
    imgL = hough_transform(roiL, imgL, 1, 1 * np.pi/180, 30, 10, 20)
    imgR = hough_transform(roiR, imgR, 1, 1 * np.pi/180, 30, 10, 20)
    result_img=cv2.hconcat([imgL,imgR])


    #윈도우 창에 비디오 재생
    cv2.imshow('original',image)
    cv2.imshow('birdeyeview',warp)
    cv2.imshow("Gray",gray_img);
    cv2.imshow("HLS",img_result);
    cv2.imshow("canny",canny_img);
    #cv2.imshow("roiLeft",roiL)
    #cv2.imshow("roiRight",roiR)
    #cv2.imshow("result",result_img)

    #종료키 설정-cv2.waitKey(time)을 이용해 time마다 프레임 재생
    if cv2.waitKey(10)&0xFF == ord('q'):
        break

#videocpature의 장치를 닫고 메모리 해제
cap.release()
#윈도우 종료
cap.destroyAllWindows()
