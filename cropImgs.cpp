#include <stdio.h>
#include <iostream>

#include <sys/types.h>
#include <dirent.h>
#include <errno.h>

#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

#include <glob.h>
#include <vector>
#include <string>

/*function... might want it in some class?*/
int getdir (string dir, vector<string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
        cout << "Error(" << errno << ") opening " << dir << endl;
        return errno;
    }

    while ((dirp = readdir(dp)) != NULL) {
        files.push_back(string(dirp->d_name));
    }
    closedir(dp);
    return 0;
}


int main(int argc, char **argv)
{
    cout <<"Usage: ./extractFrames video path/to/save/images[.]"<<endl;
    if(argc<2)        
        cout<<"Please specify the input video!"<<endl;
    char fname[50]=".";
    if(argc>2)
        sprintf(fname,"%s",argv[2]);
    
    string dir(argv[1]);
	//    string dir = string(".");
    vector<string> files = vector<string>();

    getdir(dir,files);

	Mat img;
	Rect region_of_interest = Rect(440, 181, 1028, 640);
	Mat image_roi ;
    
    char imgname[50];
    for (unsigned int i = 2;i < files.size();i++) {
        
        string name = dir + files[i];
        cout << name /*files[i] */<< endl;
        img = imread(name);
        
		image_roi = img(region_of_interest);
		sprintf(imgname, "img_%03d.jpg", i);
		imwrite(name, image_roi);
    }
    
    /*
    	g++ extractFrames.cpp `pkg-config --libs --cflags opencv` -o extractFrames

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
    }*/

    return 0;
}
