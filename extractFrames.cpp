#include <stdio.h>
#include <iostream>

#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
    cout <<"Usage: ./extractFrames video path/to/save/images[.]"<<endl;
    if(argc<2)        
        cout<<"Please specify the input video!"<<endl;
    char fname[50]=".";
    if(argc>2)
        sprintf(fname,"%s",argv[2]);
            
//    VideoCapture cap("/home/dev2/Documents/PPS_Data/PPS_SC_PBC_ISS/001/raw_pps_001_sc.avi "/*argv[1]*/);
    const char* vidname = argv[1];
    VideoCapture cap(vidname);
//    cap.open;
    if(!cap.isOpened())
{
        cout<<"Oops, cannot open the video "<<argv[1]<<"."<<endl;
return -1;
 }   
    Mat img;
    int ind=0;
    char path[100];
    while(cap.read(img))
    {
        sprintf(path,"%s/%04d.jpg",fname,ind);
	printf("%s\n",path);
        imwrite(path,img);
        ind++;
    }

    return 0;
}
