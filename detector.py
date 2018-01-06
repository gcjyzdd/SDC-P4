import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class Gradient():
    def __init__(self, sobel_kernel=3, sx_thresh=(20,100), s_thresh=(170,255)):
        self.sobel_kernel = sobel_kernel
        self.sx_thresh = sx_thresh
        self.s_thresh = s_thresh

    def setSobelKernel(self, sobel_kernel):
        self.sobel_kernel = sobel_kernel

    def setSThresh(self, s_thresh):
        self.s_thresh = s_thresh

    def setSXThresh(self,sx_thresh):
        self.sx_thresh = sx_thresh

    def preprocess(self, img):
        pass

class AbsGrad(Gradient):
    '''Calculate directional gradient'''
    def preprocess(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Apply x direction
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= self.sx_thresh[0]) & (scaled_sobel <= self.sx_thresh[1])] = 1
        return sxbinary

class MagGrad(Gradient):
    '''Calculate gradient magnitude'''
    def preprocess(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 2) Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel)
        # 3) Calculate the magnitude
        mag = np.sqrt(np.square(sobelx) + np.square(sobely))
        # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
        scaled_mag = np.uint8(255 * mag / np.max(mag))
        # 5) Create a binary mask where mag thresholds are met
        sxbinary = np.zeros_like(scaled_mag)
        sxbinary[(scaled_mag >= self.sx_thresh[0]) & (scaled_mag <= self.sx_thresh[1])] = 1
        return sxbinary

class DirGrad(Gradient):
    '''Calculate gradient direction'''
    def preprocess(self, img):
        # 1) Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 2) Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel)
        # 3) Take the absolute value of the x and y gradients
        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)
        # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
        dir = np.arctan2(abs_sobely, abs_sobelx)
        # 5) Create a binary mask where direction thresholds are met
        sxbinary = np.zeros_like(dir)
        sxbinary[(dir >= self.sx_thresh[0]) & (dir <= self.sx_thresh[1])] = 1

        # 6) Return this mask as your binary_output image
        return sxbinary

class SXGrad(Gradient):
    '''Calculate S-Channel and X directional sobel gradient'''
    def preprocess(self, img):
        # Undistortion
        dst = cv2.undistort(img, mtx, dist, None, mtx)
        # grayscale image
        gray = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
        # sobel operation applied to x axis
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        # Get absolute values
        abs_sobelx = np.absolute(sobelx)
        # Scale to (0, 255)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= self.sx_thresh[0]) & (scaled_sobel <= self.sx_thresh[1])] = 1

        # Get HLS color image
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        # Get s-channel
        s_channel = hls[:, :, 2]

        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= self.s_thresh[0]) & (s_channel <= self.s_thresh[1])] = 1

        # Combine grayscale and color gradient
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

        return combined_binary


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

    def test(self):
        print("Line prints here.")


class Detector():
    def __init__(self, mtx, dist, M, sx_thresh=(20,100), s_thresh=(170,255)):
        '''Initialization of lane line detector object'''
        # Set camera calibration and warp perspective
        self.CameraMatrix = mtx
        self.Distortion = dist
        self.WarpMatrix = M

        self.Gradient = Gradient()
        self.setBinaryFun(3)

        self.LeftLine = Line()
        self.RightLine = Line()
        # Set lane line detection uninitialized
        self.InitializedLD = False

    def setBinaryFun(self, flag=0):
        '''Set the method to generate binary gradient output of a RGB image'''
        if flag==0:
            self.Gradient = AbsGrad()
        elif flag==1:
            self.Gradient = MagGrad()
        elif flag==2:
            self.Gradient = DirGrad()
        elif flag==3:
            self.Gradient = SXGrad()
        else:
            raise 'Invalid flag:'+str(flag)

    def performBinary(self, img):
        '''Get the binary gradient output of a RGB image'''
        return self.Gradient.preprocess(img)

    def initDetection(self, binary_warped):

        pass

    def detect(self, img):
        '''Detect lane lines on an image'''

        # preprocessing img
        dst = self._undistort(img)
        binary = self.performBinary(dst)
        binary_warped = self._warp(binary)

        if self.InitializedLD:
            # we know where the lines are in the previous frame
            pass
        else:
            # Reset the detection
            self.initDetection(binary_warped)
            pass

        return binary_warped

    def visualizeDetection(self, img):
        '''Plot the detection result on the image'''

        pass

    def distance(self):
        print("The distance of two lines is .")
        return 1.0

    def _gaussian_blur(img, kernel_size=5):
        """Applies a Gaussian Noise kernel"""
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    def _undistort(self, img):
        return cv2.undistort(img, self.CameraMatrix, self.Distortion, None, self.CameraMatrix)

    def _warp(self, img):
        return cv2.warpPerspective(img, self.WarpMatrix, (img.shape[1], img.shape[0]))


# Read in the saved camera matrix and distortion coefficients
dist_pickle = pickle.load( open( "camera_cal/wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]
M = dist_pickle["M"]

a = Detector(mtx=mtx, dist=dist, M=M)
a.distance()
a.LeftLine.test()

img = mpimg.imread('test_images/straight_lines1.jpg')
tmp = a.detect(img)

plt.imshow(tmp, cmap='gray')
plt.show()

