#include <cv.h>
#include <ml.h>
#include <cxcore.h>
#include <highgui.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

#define WIDTH 30
#define HEIGHT 30

int main()
{
vector<string> src_img_path;
vector<float> src_img_catg;
unsigned int n = 0;
ifstream src_img_txt( "NN_DATA.txt" );
string buf;
while( src_img_txt )
{
   if( getline( src_img_txt, buf ) )
   {
    n++;
    if( n % 2 == 1 )
    {
     src_img_catg.push_back( atof( buf.c_str() ) );
    }
    else
    {
     src_img_path.push_back( buf );
    }
   }
}
src_img_txt.close();
int src_img_num = n / 2;

CvMat *input, *output;
input = cvCreateMat( src_img_num, WIDTH * HEIGHT, CV_32FC1 );
cvSetZero( input );
output = cvCreateMat( src_img_num, 1, CV_32FC1 );
cvSetZero( output );
IplImage *src_img, *sample_img;
float b = 0.0;
for(int i = 0; i < src_img_path.size(); i++ )
{
   src_img = cvLoadImage( src_img_path[i].c_str(), CV_LOAD_IMAGE_GRAYSCALE );
   if( src_img == NULL )
   {
    cout<<" can not load the image: "<<src_img_path[i].c_str()<<endl;
    continue;
   }
   cout<<"loading "<<src_img_path[i].c_str()<<" successfully..."<<endl;
   sample_img = cvCreateImage( cvSize(WIDTH, HEIGHT), IPL_DEPTH_8U, 1 );
   cvResize( src_img, sample_img );
   cvSmooth( sample_img, sample_img );
   n = 0;
   for(int ii = 0; ii < sample_img->height; ii++ )
   {
    for(int jj = 0; jj < sample_img->width; jj++ )
    {
     b =(float)( (int)((uchar)(sample_img->imageData + sample_img->widthStep * ii + jj )) / 255.0);
     cvmSet( input, i, n, b );
     n++;
    }
   }
   cvmSet( output, i, 0, (double)src_img_catg[i] );
   cout<<"processing "<<src_img_path[i].c_str()<<"successfully..."<<src_img_catg[i]<<endl;
}
int layer_num[3] = { WIDTH * HEIGHT, 25, 1 }; 
CvMat *layer_size = cvCreateMatHeader( 1, 3, CV_32S );
cvInitMatHeader( layer_size, 1, 3, CV_32S, layer_num ); 
CvANN_MLP nn;
nn.create( layer_size, CvANN_MLP::SIGMOID_SYM, 1, 1 );
CvANN_MLP_TrainParams params;
params.term_crit = cvTermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, \
   90000, 0.00001 );
params.train_method = 0;
params.bp_dw_scale = 0.1;
params.bp_moment_scale = 0.1;
cout<<"begin training..."<<endl;
nn.train( input, output, 0, 0, params );
cout<<"end training..."<<endl;
nn.save( "NN_DATA.xml" );

ifstream tst_txt( "NN_TEST.txt" );
vector<string> tst_path;
while( tst_txt )
{
   if( getline( tst_txt, buf ) )
   { 
    tst_path.push_back( buf );
   }
}
tst_txt.close();
CvMat* tst_mat = cvCreateMat( 1, WIDTH * HEIGHT, CV_32FC1 );
CvMat* tst_res = cvCreateMat( 1, 1, CV_32FC1 );
IplImage *tst_img, *tmp_img;
ofstream tst_output_txt( "NN_PREDICT.txt" );
char line[512];
for(int j = 0; j < tst_path.size(); j++)
{
   tst_img = cvLoadImage( tst_path[j].c_str(), CV_LOAD_IMAGE_GRAYSCALE );
   if( tst_img == NULL )
   {
    cout<<" can not load the image: "<<tst_path[j].c_str()<<endl;
    continue;
   }
   tmp_img = cvCreateImage( cvSize(WIDTH, HEIGHT), IPL_DEPTH_8U, 1 );
   cvResize( tst_img, tmp_img );
   cvSmooth( tmp_img, tmp_img );
   n = 0;
   for( int ii = 0; ii < tmp_img->height; ii++ )
   {
    for( int jj = 0; jj < tmp_img->width; jj++ )
    {
     b = (int)( (uchar)(tmp_img->imageData + tmp_img->widthStep * ii + jj ) ) / 255.0;
     cvmSet( tst_mat, 0, n, b );
    }
   }
   nn.predict( tst_mat, tst_res );
   sprintf( line, "%s %f\r\n", tst_path[j].c_str(), cvmGet(tst_res,0,0) );
   tst_output_txt<<line;
}
tst_output_txt.close();
return 0;
}