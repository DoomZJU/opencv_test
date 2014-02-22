#include <stdio.h>
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <iostream>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include "core/core.hpp"
#include "features2d/features2d.hpp"
#include "highgui/highgui.hpp"
#include "nonfree/features2d.hpp"
#include <nonfree/nonfree.hpp>  

#define MAX_CORNERS 100
#define PI 3.14

using namespace std;
using namespace cv;

int main( int argc, char** argv )
{

  //����IplImageָ��
  IplImage* pFrame = NULL;

 //��ȡ����ͷ
  CvCapture* pCapture = cvCreateCameraCapture(0);
 
  //��������
  cvNamedWindow("���������ͷ", 1);
 
  //��ʾ����
  while(1)
  {
     pFrame = cvQueryFrame( pCapture );
     if(!pFrame)break;
     cvShowImage("���������ͷ",pFrame);

//------------ֱ��ͼ--------------------------------------------------//
	 //cvNamedWindow( "H-S Histogram", 1 );
	 //IplImage* hsv = cvCreateImage( cvGetSize(pFrame), 8, 3 );
  //   IplImage* h_plane = cvCreateImage( cvGetSize(pFrame), 8, 1 );
  //   IplImage* s_plane = cvCreateImage( cvGetSize(pFrame), 8, 1 );
  //   IplImage* v_plane = cvCreateImage( cvGetSize(pFrame), 8, 1 );
  //   IplImage* planes[] = { h_plane, s_plane };
 
  //   /** H ��������Ϊ16���ȼ���S��������Ϊ8���ȼ�*/
  //   int h_bins = 16, s_bins = 8;
  //   int hist_size[] = {h_bins, s_bins};
 
  //   /** H �����ı仯��Χ*/
  //   float h_ranges[] = { 0, 180 };
 
  //   /** S �����ı仯��Χ*/
  //   float s_ranges[] = { 0, 255 };
  //   float* ranges[] = { h_ranges, s_ranges };
 
  //   /** ����ͼ��ת����HSV��ɫ�ռ�*/
  //   cvCvtColor( pFrame, hsv, CV_BGR2HSV );
  //   cvCvtPixToPlane( hsv, h_plane, s_plane, v_plane, 0 );
 
  //   /** ����ֱ��ͼ����ά, ÿ��ά���Ͼ���*/
  //   CvHistogram * hist = cvCreateHist( 2, hist_size, CV_HIST_ARRAY, ranges, 1 );
  //   /** ����H,S����ƽ������ͳ��ֱ��ͼ*/
  //   cvCalcHist( planes, hist, 0, 0 );
 
  //   /** ��ȡֱ��ͼͳ�Ƶ����ֵ�����ڶ�̬��ʾֱ��ͼ*/
  //   float max_value;
  //   cvGetMinMaxHistValue( hist, 0, &max_value, 0, 0 );
 
 
  //   /** ����ֱ��ͼ��ʾͼ��*/
  //   int height = 240;
  //   int width = (h_bins*s_bins*6);
  //   IplImage* hist_img = cvCreateImage( cvSize(width,height), 8, 3 );
  //   cvZero( hist_img );
 
  //   /** ��������HSV��RGB��ɫת������ʱ��λͼ��*/
  //   IplImage * hsv_color = cvCreateImage(cvSize(1,1),8,3);
  //   IplImage * rgb_color = cvCreateImage(cvSize(1,1),8,3);
  //   int bin_w = width / (h_bins * s_bins);
  //   for(int h = 0; h < h_bins; h++)
  //   {
  //       for(int s = 0; s < s_bins; s++)
  //       {
  //            int i = h*s_bins + s;
  //            /** ���ֱ��ͼ�е�ͳ�ƴ�����������ʾ��ͼ���еĸ߶�*/
  //            float bin_val = cvQueryHistValue_2D( hist, h, s );
  //            int intensity = cvRound(bin_val*height/max_value);
 
  //            /** ��õ�ǰֱ��ͼ�������ɫ��ת����RGB���ڻ���*/
  //            cvSet2D(hsv_color,0,0,cvScalar(h*180.f / h_bins,s*255.f/s_bins,255,0));
  //            cvCvtColor(hsv_color,rgb_color,CV_HSV2BGR);
  //            CvScalar color = cvGet2D(rgb_color,0,0);
 
  //            cvRectangle( hist_img, cvPoint(i*bin_w,height),
  //                 cvPoint((i+1)*bin_w,height - intensity),
  //                 color, -1, 8, 0 );
  //       }
  //   }

  //   cvShowImage( "H-S Histogram", hist_img );

//------------ֱ��ͼ--------------------------------------------------//

//------------Canny--------------------------------------------------//
	 ////����IplImageָ��
  //       IplImage* pFrame1 = cvCreateImage(cvGetSize(pFrame),pFrame->depth,1);
  //       IplImage* pCannyImg = NULL;
  //       //����ͼ��ǿ��ת��ΪGray
  //       //pImg = cvLoadImage( "E:\\Download\\test.jpg", 0);
		// cvCvtColor(pFrame,pFrame1,CV_BGR2GRAY);
  //       //Ϊcanny��Եͼ������ռ�
  //       pCannyImg = cvCreateImage(cvGetSize(pFrame1), IPL_DEPTH_8U, 1);
  //       //canny��Ե���
  //       cvCanny(pFrame1, pCannyImg, 50, 150, 3);
  //       //��������
  //       cvNamedWindow("canny",1);
  //       //��ʾͼ��
  //       cvShowImage( "canny", pCannyImg );

//------------Canny-------------------------------------------------//

//------------�ǵ�-------------------------------------------------//
	//int cornersCount=MAX_CORNERS;//�õ��Ľǵ���Ŀ
	//CvPoint2D32f corners[MAX_CORNERS];//����ǵ㼯��
	//IplImage *grayImage = NULL,*corners1 = NULL,*corners2 = NULL;
	//int i;
	//CvScalar color = CV_RGB(255,0,0);
 //
	////Load the image to be processed
	////srcImage = cvLoadImage("E:\\Download\\1.jpg",1);
	//grayImage = cvCreateImage(cvGetSize(pFrame),IPL_DEPTH_8U,1);
 //
	////copy the source image to copy image after converting the format
	////���Ʋ�תΪ�Ҷ�ͼ��
	//cvCvtColor(pFrame,grayImage,CV_BGR2GRAY);
 //
	////create empty images os same size as the copied images
	////������ʱλ����ͼ��cvGoodFeaturesToTrack���õ�
	//corners1 = cvCreateImage(cvGetSize(pFrame),IPL_DEPTH_32F,1);
	//corners2 = cvCreateImage(cvGetSize(pFrame),IPL_DEPTH_32F,1);
 //
	//cvGoodFeaturesToTrack(grayImage,corners1,corners2,corners,&cornersCount,0.05,
	//30,//�ǵ����С������
	//0,//����ͼ��
	//3,0,0.4);
	//printf("num corners found: %d\n",cornersCount);
 //
	////��ʼ����ÿ����
	//if (cornersCount>0)
	//{
	//for (i=0;i<cornersCount;i++)
	//{
	//cvCircle(pFrame,cvPoint((int)(corners[i].x),(int)(corners[i].y)),2,color,2,CV_AA,0);
	//}
	//}
	//cvShowImage("���������ͷ",pFrame);
 
	//cvReleaseImage(&pFrame);
	//cvReleaseImage(&grayImage);
	//cvReleaseImage(&corners1);
	//cvReleaseImage(&corners2);

//------------�ǵ�-------------------------------------------------//

//-----------Houghֱ��---------------------------------------------//

    /*IplImage* dst;
    IplImage* color_dst;
    CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq* lines = 0;
    int i;
 
    dst = cvCreateImage( cvGetSize(pFrame), 8, 1 );
    color_dst = cvCreateImage( cvGetSize(pFrame), 8, 3 );
 
    cvCanny( pFrame, dst, 50, 200, 3 );
    cvCvtColor( dst, color_dst, CV_GRAY2BGR );
#if 0
    lines = cvHoughLines2( dst, storage, CV_HOUGH_STANDARD, 1, CV_PI/180, 100, 0, 0 );
 
    for( i = 0; i < MIN(lines->total,100); i++ )
    {
        float* line = (float*)cvGetSeqElem(lines,i);
        float rho = line[0];
        float theta = line[1];
        CvPoint pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));
        cvLine( color_dst, pt1, pt2, CV_RGB(255,0,0), 3, CV_AA, 0 );
    }
#else
    lines = cvHoughLines2( dst, storage, CV_HOUGH_PROBABILISTIC, 1, CV_PI/180, 50, 50, 10 );
    for( i = 0; i < lines->total; i++ )
    {
        CvPoint* line = (CvPoint*)cvGetSeqElem(lines,i);
        cvLine( color_dst, line[0], line[1], CV_RGB(255,0,0), 3, CV_AA, 0 );
    }
#endif
    cvShowImage( "���������ͷ", pFrame );
 
    cvNamedWindow( "Hough", 1 );
    cvShowImage( "Hough", color_dst );*/

//-----------Houghֱ��---------------------------------------------//

//-----------HoughԲ----------------------------------------------//

// IplImage* gray = cvCreateImage( cvGetSize(pFrame), 8, 1 );
//     CvMemStorage* storage = cvCreateMemStorage(0);
//     cvCvtColor( pFrame, gray, CV_BGR2GRAY );
//     cvSmooth( gray, gray, CV_GAUSSIAN, 5, 15 );
//// smooth it, otherwise a lot of false circles may be detected
//CvSeq* circles = cvHoughCircles( gray, storage, CV_HOUGH_GRADIENT, 2, gray->height/4, 200, 100 );
//    int i;
//     for( i = 0; i < circles->total; i++ )
//     {
//          float* p = (float*)cvGetSeqElem( circles, i );
//          cvCircle( pFrame, cvPoint(cvRound(p[0]),cvRound(p[1])), 3, CV_RGB(0,255,0), -1, 8, 0 );
// cvCircle( pFrame, cvPoint(cvRound(p[0]),cvRound(p[1])), cvRound(p[2]), CV_RGB(255,0,0), 3, 8, 0 );
//          cout<<"Բ������x= "<<cvRound(p[0])<<endl<<"Բ������y= "<<cvRound(p[1])<<endl;
//          cout<<"�뾶="<<cvRound(p[2])<<endl;
//     }
//     cout<<"Բ����="<<circles->total<<endl;
//     cvShowImage( "���������ͷ", pFrame );

//-----------HoughԲ----------------------------------------------//

//-----------��Եֱ��ͼ----------------------------------------------//
 
//    IplImage *histimg = 0; // histogram image
//    CvHistogram *hist = 0; // define multi_demention histogram
//    IplImage* canny;
//    CvMat* canny_m;
//    IplImage* dx; // the sobel x difference
//    IplImage* dy; // the sobel y difference
//    CvMat* gradient; // value of gradient
//    CvMat* gradient_dir; // direction of gradient
//    CvMat* dx_m; // format transform to matrix
//    CvMat* dy_m;
//    CvMat* mask;
//    CvSize  size;
//    IplImage* gradient_im;
//    int i,j;
//    float theta;
//   
//    int hdims = 8;     // ����HIST�ĸ�����Խ��Խ��ȷ
//    float hranges_arr[] = {-PI/2,PI/2}; // ֱ��ͼ���Ͻ���½�
//    float* hranges = hranges_arr;
//                                                                                                                                                                                                                                                              
//    float max_val;  //
//    int bin_w;
//   
//    cvNamedWindow( "Histogram", 0 );
//    //cvNamedWindow( "src", 0);
//    size=cvGetSize(pFrame);
//    canny=cvCreateImage(cvGetSize(pFrame),8,1);//��Եͼ��
//    dx=cvCreateImage(cvGetSize(pFrame),32,1);//x�����ϵĲ��//�˴�����������ΪU ���������
//    dy=cvCreateImage(cvGetSize(pFrame),32,1);
//    gradient_im=cvCreateImage(cvGetSize(pFrame),32,1);//�ݶ�ͼ��
//    canny_m=cvCreateMat(size.height,size.width,CV_32FC1);//��Ե����
//    dx_m=cvCreateMat(size.height,size.width,CV_32FC1);
//    dy_m=cvCreateMat(size.height,size.width,CV_32FC1);
//    gradient=cvCreateMat(size.height,size.width,CV_32FC1);//�ݶȾ���
//    gradient_dir=cvCreateMat(size.height,size.width,CV_32FC1);//�ݶȷ������
//    mask=cvCreateMat(size.height,size.width,CV_32FC1);//����
// 
//    cvCanny(pFrame,canny,60,180,3);//��Ե���
//    cvConvert(canny,canny_m);//��ͼ��ת��Ϊ����
//    cvSobel(pFrame,dx,1,0,3);// һ��X�����ͼ����:dx
//    cvSobel(pFrame,dy,0,1,3);// һ��Y�����ͼ����:dy
//    cvConvert(dx,dx_m);
//    cvConvert(dy,dy_m);
//    cvAdd(dx_m,dy_m,gradient); // value of gradient//�ݶȲ��ǵ��ڸ�����x�ĵ�����ƽ������y������ƽ����
//    cvDiv(dx_m,dy_m,gradient_dir); // direction
//    for(i=0;i<size.height;i++)
//    for(j=0;j<size.width;j++)
//    {
//      if(cvmGet(canny_m,i,j)!=0 && cvmGet(dx_m,i,j)!=0)//������ʲô��˼��ֻ����Ե�ϵķ���
//      {
//         theta=cvmGet(gradient_dir,i,j);
//         theta=atan(theta);
//         cvmSet(gradient_dir,i,j,theta); 
//      }
//      else
//      {
//         cvmSet(gradient_dir,i,j,0);
//      }
//        
//    }
//   hist = cvCreateHist( 1, &hdims, CV_HIST_ARRAY, &hranges, 1 ); 
//// ����һ��ָ���ߴ��ֱ��ͼ�������ش�����ֱ��ͼָ��
//   histimg = cvCreateImage( cvSize(320,200), 8, 3 ); // ����һ��ͼ��ͨ��
//   cvZero( histimg ); // �壻
//   cvConvert(gradient_dir,gradient_im);//���ݶȷ������ת��Ϊͼ��
//   cvCalcHist( &gradient_im, hist, 0, canny ); // ����ֱ��ͼ
//   cvGetMinMaxHistValue( hist, 0, &max_val, 0, 0 );  // ֻ�����ֵ
//   cvConvertScale( hist->bins, hist->bins, max_val ? 255. / max_val : 0., 0 );
//// ����bin ������[0,255] ������ϵ��
//   cvZero( histimg );
//   bin_w = histimg->width /16;  // hdims: ���ĸ�������bin_w Ϊ���Ŀ��
//   
//    // ��ֱ��ͼ
//    for( i = 0; i < hdims; i++ )
//    {
//       double val = ( cvGetReal1D(hist->bins,i)*histimg->height/255 );
//// ���ص�ͨ�������ָ��Ԫ�أ� ����ֱ��ͼ��i���Ĵ�С��valΪhistimg�е�i���ĸ߶�
//        CvScalar color = CV_RGB(255,255,0); //(hsv2rgb(i*180.f/hdims);//ֱ��ͼ��ɫ
//        cvRectangle( histimg, cvPoint(100+i*bin_w,histimg->height),cvPoint(100+(i+1)*bin_w,(int)(histimg->height - val)), color, 1, 8, 0 ); // ��ֱ��ͼ���������Σ����½ǣ����Ͻ�����
//     }
//   
//    cvShowImage( "���������ͷ", pFrame);
//    cvShowImage( "Histogram", histimg );

//-----------��Եֱ��ͼ----------------------------------------------//

//-----------������ȡ----------------------------------------------//

	/*int tmp[8]={0};
	int sum=0;int k=0;
	IplImage *dst;
	CvScalar s;
	cvNamedWindow("dst",1);
	cvShowImage("���������ͷ",pFrame);
 
	uchar* data=(uchar*)pFrame->imageData;
	int step=pFrame->widthStep;
	dst=cvCreateImage(cvSize(pFrame->width,pFrame->height),pFrame->depth,1);
	dst->widthStep=pFrame->widthStep;
	for(int i=1;i<pFrame->height-1;i++)
	for(int j=1;j<pFrame->width-1;j++)
	{
	if(data[(i-1)*step+j-1]>data[i*step+j]) tmp[0]=1;
	else tmp[0]=0;
	if(data[i*step+(j-1)]>data[i*step+j]) tmp[1]=1;
	else tmp[1]=0;
	if(data[(i+1)*step+(j-1)]>data[i*step+j]) tmp[2]=1;
	else tmp[2]=0;
	if (data[(i+1)*step+j]>data[i*step+j]) tmp[3]=1;
	else tmp[3]=0;
	if (data[(i+1)*step+(j+1)]>data[i*step+j]) tmp[4]=1;
	else tmp[4]=0;
	if(data[i*step+(j+1)]>data[i*step+j]) tmp[5]=1;
	else tmp[5]=0;
	if(data[(i-1)*step+(j+1)]>data[i*step+j]) tmp[6]=1;
	else tmp[6]=0;
	if(data[(i-1)*step+j]>data[i*step+j]) tmp[7]=1;
	else tmp[7]=0;
	for(k=0;k<=7;k++)
	sum+=abs(tmp[k]-tmp[k+1]);
	sum=sum+abs(tmp[7]-tmp[0]);
	if (sum<=2)
	s.val[0]=(tmp[0]*128+tmp[1]*64+tmp[2]*32+tmp[3]*16+tmp[4]*8+tmp[5]*4+tmp[6]*2+tmp[7]);
	else s.val[0]=59;
	cvSet2D(dst,i,j,s);
	}
 
	cvShowImage("dst",dst);*/

//-----------������ȡ----------------------------------------------//

//-----------SURF-------------------------------------------------//

  IplImage* pFrame1 = cvCreateImage(cvGetSize(pFrame),pFrame->depth,1);
  cvCvtColor(pFrame,pFrame1,CV_BGR2GRAY);

  Mat mat = pFrame1;

  //-- Step 1: Detect the keypoints using SURF Detector
  int minHessian = 400;

  SurfFeatureDetector detector( minHessian );

  std::vector<KeyPoint> keypoints;

  detector.detect( mat, keypoints );

  //-- Draw keypoints
  Mat img_keypoints;

  drawKeypoints( mat, keypoints, img_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

  //-- Show detected (drawn) keypoints
  imshow("Keypoints 1", img_keypoints );

//-----------SURF-------------------------------------------------//

     char c=cvWaitKey(33);
     if(c==27)break;
  }
  cvReleaseCapture(&pCapture);
  //cvDestroyWindow("���������ͷ");
  cvDestroyAllWindows();
}