import pickle
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.ndimage

from multiprocessing.pool import ThreadPool

import time
from functools import wraps

### define profilers (https://stackoverflow.com/questions/3620943/measuring-elapsed-time-with-the-time-module)
PROF_DATA = {}

def profile(fn):
    @wraps(fn)
    def with_profiling(*args, **kwargs):
        start_time = time.time()

        ret = fn(*args, **kwargs)

        elapsed_time = time.time() - start_time

        if fn.__name__ not in PROF_DATA:
            PROF_DATA[fn.__name__] = [0, []]
        PROF_DATA[fn.__name__][0] += 1
        PROF_DATA[fn.__name__][1].append(elapsed_time)

        return ret

    return with_profiling

def print_prof_data():
    for fname, data in PROF_DATA.items():
        max_time = max(data[1])
        avg_time = sum(data[1]) / len(data[1])
        print("Function %s called %d times. " % (fname, data[0]))
        print('Execution time max: %.3f, average: %.3f' % (max_time, avg_time))

def clear_prof_data():
    global PROF_DATA
    PROF_DATA = {}


### end of profiler

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
    """Calculate directional gradient"""
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
    """Calculate gradient magnitude"""
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
    """Calculate gradient direction"""
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
    """Calculate S-Channel and X directional sobel gradient"""
    def preprocess(self, img):
        # grayscale image
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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


class SChannelGrad(Gradient):

    def preprocess(self, img):
        # Get HLS color image
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        # Get s-channel
        s_channel = hls[:, :, 2]

        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= self.s_thresh[0]) & (s_channel <= self.s_thresh[1])] = 1

        return s_binary


class LightAutoGrad(Gradient):

    def preprocess(self, img):
        # grayscale image
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        mean_gray = np.mean(gray)

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

        if mean_gray > 80:
            # Get s-channel
            s_channel = hls[:, :, 2]
        else:# Dark emvironment
            # Get l-channel
            s_channel = hls[:, :, 1]

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
        # Save n fits
        self.N = 5
        # Margin
        self.margin = 50
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients of the last n iterations
        self.fits = []
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # save polynomial coeffs of last iteration
        self.last_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

        self.MaxFail = 5
        self.fail_num = 0

    def get_init_xy(self, base, binary_warped):
        """Get initial valid pixel coordinates"""
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        x_current = base
        # Set the width of the windows +/- margin
        margin = self.margin
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_x_low = x_current - margin
            win_x_high = x_current + margin

            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_x_low, win_y_low), (win_x_high, win_y_high),
                          (0, 255, 0), 2)

            # Identify the nonzero pixels in x and y within the window
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]

            # Append these indices to the lists
            lane_inds.append(good_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_inds) > minpix:
                x_current = np.int(np.mean(nonzerox[good_inds]))

        # Concatenate the arrays of indices
        lane_inds = np.concatenate(lane_inds)

        # Extract left and right line pixel positions
        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]

        self.allx = x
        self.ally = y

    def get_ctn_xy(self, binary_warped):
        """Get valid pixel coordinates from previous detection"""
        # Update last fit
        self.last_fit = self.current_fit

        # we know where the lines are in the previous frame
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = self.margin

        fit = self.current_fit
        lane_inds = ((nonzerox > (fit[0] * (nonzeroy ** 2) + fit[1] * nonzeroy +
                                  fit[2] - margin)) & (nonzerox < (fit[0] * (nonzeroy ** 2) +
                                                                   fit[1] * nonzeroy + fit[
                                                                                 2] + margin)))
        # Extract left and right line pixel positions
        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]

        self.allx = x
        self.ally = y

    def setNum(self, n):
        self.N = n

    def test(self):
        print("Line prints here.")


class KalmanFilter():
    """Implement a simple Kalman filter"""
    def __init__(self, n_in, q=(1,1,1), R=(1,1,1)):
        self.P = np.array(q)#np.array([[q[0],0,0],[0,q[1],0],[0,0,q[2]]], np.float64)
        self.R = np.array(R)#np.array([[R[0],0,0],[0,R[1],0],[0,0,R[2]]], np.float64)
        self.K = None
        self.P_pr = None
        self.X = np.zeros((n_in, ), np.float64)

        self.Init = False

    def update(self, X_e):

        if not self.Init:
            self.Init = True
            self.X = X_e
            self.X_pr = self.X
            self.P_pr = self.P
            return self.X.flatten()

        for i in range(3):
            xe = X_e[i]
            self.X_pr[i] = self.X[i]
            K = self.P_pr[i]/(self.P_pr[i]+self.R[i])
            self.X[i] = self.X_pr[i]+K*(X_e[i]-self.X_pr[i])
            self.P[i] = (1-K)*self.P_pr[i]

        return self.X.flatten()

        X_e = X_e[:,None]
        # time update
        self.X_pr = self.X
        self.P_pr = self.P

        # time update
        self.K = self.P_pr * np.linalg.inv(self.P_pr + self.R)
        self.X = self.X_pr + np.matmul(self.K, (X_e-self.X_pr))
        self.P = np.matmul((np.eye(3)-self.K), self.P_pr)

        return self.X.flatten()

    def updatebk(self, X_e):

        if ~self.Init:
            self.Init = True
            self.X = X_e[:, None]
            self.X_pr = self.X
            return self.X.flatten()

        X_e = X_e[:,None]
        # time update
        self.X_pr = self.X
        self.P_pr = self.P

        # time update
        self.K = np.matmul(self.P_pr, np.linalg.inv(self.P_pr + self.R))
        self.X = self.X_pr + np.matmul(self.K, (X_e-self.X_pr))
        self.P = np.matmul((np.eye(3)-self.K), self.P_pr)

        return self.X.flatten()

    def setPR(self,P,R):
        self.P = P
        self.R = R

    def printState(self):
        print('X=', self.X)


class Detector():
    def __init__(self, mtx, dist, M, Minv, sx_thresh=(20,100), s_thresh=(170,255)):
        """Initialization of lane line detector object"""
        # Set camera calibration and warp perspective
        self.CameraMatrix = mtx
        self.Distortion = dist
        self.WarpMatrix = M
        self.WarpMatrixInv = Minv

        self.Gradient = Gradient()
        self.setBinaryFun(3)

        self.LeftLine = Line()
        self.RightLine = Line()

        self.KFLeft = KalmanFilter(3, q=(4e-8, 1e-2, 100), R=(1e-4, 1, 10000))
        self.KFRight = KalmanFilter(3, q=(4e-8, 1e-2, 100), R=(1e-4, 1, 10000))
        self.UseKalmanFilter = True

        # Set lane line detection uninitialized
        self.InitializedLD = False

        # cache the states of last 5 iterations
        self.cacheNum = 5

        self.ploty = None
        self.distTop = 0
        self.dist_to_left = 0
        self.distButtom = 0
        self.img = None
        self.undist = None

        self.margin = 50
        # Define conversions in x and y from pixels space to meters
        self.ym_per_pix = 30 / 720  # meters per pixel in y dimension
        self.xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        self.FitLeft = []
        self.FitRight = []

        self.debug = False
        # self.pool = ThreadPool(processes=1)

    def setMargin(self,margin):
        self.margin = margin

    def switchKF(self, status):
        self.UseKalmanFilter = status

    def setMaxFail(self, mf):
        self.LeftLine.MaxFail = mf
        self.RightLine.MaxFail = mf

    def setBinaryFun(self, flag=0):
        """Set the method to generate binary gradient output of a RGB image"""
        if flag==0:
            self.Gradient = AbsGrad()
        elif flag==1:
            self.Gradient = MagGrad()
        elif flag==2:
            self.Gradient = DirGrad()
        elif flag==3:
            self.Gradient = SXGrad()
        elif flag==4:
            self.Gradient = SChannelGrad()
        elif flag==5:
            self.Gradient = LightAutoGrad()
        else:
            raise 'Invalid flag:'+str(flag)

    @profile
    def performBinary(self, img):
        """Get the binary gradient output of a RGB image"""

        return self.Gradient.preprocess(img)

    def get_xy_pvalue(self, binary_warped):
        """Get the xy pixel values for fitting"""
        if self.LeftLine.detected:
            self.LeftLine.get_ctn_xy(binary_warped)
        else:
            ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
            self.ploty = ploty

            histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
            midpoint = np.int(histogram.shape[0] / 2)
            leftx_base = np.argmax(histogram[:midpoint])
            self.LeftLine.get_init_xy(leftx_base, binary_warped)

        if self.RightLine.detected:
            self.RightLine.get_ctn_xy(binary_warped)
        else:
            histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
            midpoint = np.int(histogram.shape[0] / 2)
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint
            self.RightLine.get_init_xy(rightx_base, binary_warped)

    @profile
    def detect(self, img):
        """Detect lane lines on an image"""

        self.img = img
        # preprocessing img
        self.undist = self._undistort(img)
        #binary = self.performBinary(self.undist)
        #binary_warped = self._warp(binary)

        warped = self._warp(self.undist)

        if self.debug:
            db_gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
            print('Mean gray = ', np.mean(db_gray))

        binary_warped = self.performBinary(warped)
        binary_warped = self._gaussian_blur(binary_warped)

        # Get valid pixel coordinates
        self.get_xy_pvalue(binary_warped)

        # Fit a second order polynomial to each
        left_fit = np.polyfit(self.LeftLine.ally, self.LeftLine.allx, 2)
        right_fit = np.polyfit(self.RightLine.ally, self.RightLine.allx, 2)

        self.sanityCheck(left_fit, right_fit)
        self.update()

        output = self.visualizeInput()
        binOut = self.visualizeDetection(binary_warped)

        # height = output.shape[0]
        width = output.shape[1]
        scaleDown = 0.4
        height_s = math.floor(binOut.shape[0]*scaleDown)
        width_s = math.floor(binOut.shape[1]*scaleDown)
        #binOut = self._zoomImg(binOut, (scaleDown,scaleDown,1))# extremely slow
        binOut = cv2.resize(binOut, (width_s, height_s))
        output[0:height_s, (width-width_s):width,:] = binOut

        return output#self.visualizeInput()

    @profile
    def visualizeDetection(self, img):
        """Plot the detection result on the warped binary image"""
        # Create an image to draw on and an image to show the selection window
        if len(img.shape) > 2:
            out_img = np.copy(img)
        else:
            out_img = np.dstack((img, img, img)) * 255

        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        #out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        #out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        left_fit = self.LeftLine.current_fit
        right_fit = self.RightLine.current_fit
        ploty = self.ploty
        left_fitx = self.LeftLine.recent_xfitted[-1]
        right_fitx = self.RightLine.recent_xfitted[-1]
        margin = self.margin

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                        ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                         ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        result[ploty.astype(np.int32), left_fitx.astype(np.int32)] = [255, 0, 0]
        result[ploty.astype(np.int32), left_fitx.astype(np.int32)-1] = [255, 0, 0]
        result[ploty.astype(np.int32), left_fitx.astype(np.int32)+1] = [255, 0, 0]
        result[ploty.astype(np.int32), right_fitx.astype(np.int32)] = [255, 0, 0]
        result[ploty.astype(np.int32), right_fitx.astype(np.int32)-1] = [255, 0, 0]
        result[ploty.astype(np.int32), right_fitx.astype(np.int32)+1] = [255, 0, 0]

        if self.debug:
            plt.imshow(result)
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')
            plt.xlim(0, 1280)
            plt.ylim(720, 0)
            plt.show()

        return result

    @profile
    def visualizeInput(self):
        """Plot the result on the input RGB image"""

        # Create an image to draw the lines on
        color_warp = np.zeros_like(self.img).astype(np.uint8)
        #color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        left_fitx = self.LeftLine.recent_xfitted[-1]
        right_fitx = self.RightLine.recent_xfitted[-1]
        ploty = self.ploty
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = self._invwarp(color_warp)
        # Combine the result with the original image
        result = cv2.addWeighted(self.undist, 1, newwarp, 0.3, 0)

        cv2.putText(result, "Radius of curvature = {:4d}m".format(math.floor(self.LeftLine.radius_of_curvature)),
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, thickness=2)
        cv2.putText(result, "Distance to left line = {:3.2f}m".format(self.dist_to_left),
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, thickness=2)

        return result.astype(np.uint8)

    @profile
    def getCurvature(self, left_fit, right_fit):
        """Calculate curvature of two lines"""
        # Fit new polynomials to x,y in world space
        ploty = self.ploty
        ym_per_pix = self.ym_per_pix
        xm_per_pix = self.xm_per_pix

        leftx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        rightx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        y_eval = np.max(ploty)

        left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / \
                        np.absolute(2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / \
                         np.absolute(2 * right_fit_cr[0])

        return left_curverad, right_curverad

    def get_dist_to_left(self):
        xm_per_pix = self.xm_per_pix
        dist_to_left = xm_per_pix * (self.img.shape[1] / 2 - self.LeftLine.recent_xfitted[-1][-1])

        return dist_to_left

    def distance(self):
        print("The distance of two lines is .")
        return 1.0

    def sanityCheck(self, left_fit, right_fit):

        self.LeftLine.diffs = left_fit - self.LeftLine.current_fit
        self.RightLine.diffs = right_fit - self.RightLine.current_fit

        curvts = self.getCurvature(left_fit, right_fit)
        curvts = np.absolute(curvts)
        ratio = np.max(curvts)/np.min(curvts)

        ploty = self.ploty
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        dmax = np.max(np.absolute(left_fitx - right_fitx))
        dmin = np.min(np.absolute(left_fitx - right_fitx))

        if ratio > 5 or (dmax-dmin)/dmin > 0.3:
            self.LeftLine.fail_num += 1
            if self.LeftLine.fail_num == self.LeftLine.MaxFail:
                # reset detection if fails MaxFail times in a row
                self.InitializedLD = False
                self.LeftLine.detected = False
                self.RightLine.detected = False
        else:
            self.InitializedLD = True
            self.LeftLine.detected = True
            self.RightLine.detected = True

            self.LeftLine.fail_num = 0

            self.LeftLine.current_fit = left_fit
            self.RightLine.current_fit = right_fit

            # Pushback states
            if len(self.LeftLine.fits) < self.LeftLine.N:
                self.LeftLine.fits.append(self.LeftLine.current_fit)
            else:
                self.LeftLine.fits.pop(0)
                self.LeftLine.fits.append(self.LeftLine.current_fit)

            if len(self.RightLine.fits) < self.RightLine.N:
                self.RightLine.fits.append(self.RightLine.current_fit)
            else:
                self.RightLine.fits.pop(0)
                self.RightLine.fits.append(self.RightLine.current_fit)

            if len(self.LeftLine.recent_xfitted) < self.LeftLine.N:
                self.LeftLine.recent_xfitted.append(left_fitx)
            else:
                self.LeftLine.recent_xfitted.pop(0)
                self.LeftLine.recent_xfitted.append(left_fitx)

            if len(self.RightLine.recent_xfitted) < self.RightLine.N:
                self.RightLine.recent_xfitted.append(right_fitx)
            else:
                self.RightLine.recent_xfitted.pop(0)
                self.RightLine.recent_xfitted.append(right_fitx)

    def update(self):
        # Smooth the result
        self.LeftLine.current_fit = np.mean(np.array(self.LeftLine.fits), axis=0)#self.LeftLine.last_fit
        self.RightLine.current_fit = np.mean(np.array(self.RightLine.fits), axis=0)  # self.RightLine.last_fit

        curvts = self.getCurvature(self.LeftLine.current_fit, self.RightLine.current_fit)
        self.LeftLine.radius_of_curvature = curvts[0]
        self.RightLine.radius_of_curvature = curvts[1]
        self.dist_to_left = self.get_dist_to_left()

    def setKF_PR(self,P,R):
        self.KFLeft.setPR(P, R)
        self.KFRight.setPR(P,R)

    def _gaussian_blur(self, img, kernel_size=5):
        """Applies a Gaussian Noise kernel"""
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    @profile
    def _undistort(self, img):
        """Apply image undistortion"""
        return cv2.undistort(img, self.CameraMatrix, self.Distortion, None, self.CameraMatrix)

    @profile
    def _warp(self, img):
        """Apply image warp transformation"""
        return cv2.warpPerspective(img, self.WarpMatrix, (img.shape[1], img.shape[0]))

    def _invwarp(self, img):
        """Apply image inverse warp transformation"""
        return cv2.warpPerspective(img, self.WarpMatrixInv, (img.shape[1], img.shape[0]))

    @profile
    def _zoomImg(self, img, scale):
        return scipy.ndimage.zoom(img, scale)

    def plotFit(self):
        self._plotFit(self.FitLeft)
        self._plotFit(self.FitRight)

    def _plotFit(self, fits):
        x = np.array(fits)
        L = x.shape[0]
        t = np.arange(0,L,1)

        plt.figure(1)
        plt.subplot(311)
        plt.plot(t, x[:, 0])
        plt.title('fit[0]')
        plt.grid(True)

        plt.subplot(312)
        plt.plot(t, x[:, 1])
        plt.title('fit[1]')
        plt.grid(True)

        plt.subplot(313)
        plt.plot(t, x[:, 2])
        plt.title('fit[2]')
        plt.grid(True)

        plt.show()

    def initDetection(self, binary_warped):
        """Initialize the detection"""
        self.InitializedLD = True
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = self.margin
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                          (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                          (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Update states of left and right lines
        self.LeftLine.current_fit = left_fit
        self.LeftLine.line_base_pos = leftx_base
        self.LeftLine.allx = leftx
        self.LeftLine.ally = lefty

        self.RightLine.current_fit = right_fit
        self.RightLine.line_base_pos = rightx_base
        self.RightLine.allx = rightx
        self.RightLine.ally = righty

        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        self.ploty = ploty

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        self.ploty = ploty
        self.LeftLine.recent_xfitted = left_fitx
        self.RightLine.recent_xfitted = right_fitx

        self.distTop = np.absolute(left_fitx[0] - right_fitx[0])
        self.distButtom = np.absolute(left_fitx[-1] - right_fitx[0])

        # Pushback states
        if len(self.LeftLine.fits) < self.LeftLine.N:
            self.LeftLine.fits.append(self.LeftLine.current_fit)
        else:
            self.LeftLine.fits.pop(0)
            self.LeftLine.fits.append(self.LeftLine.current_fit)

        if len(self.RightLine.fits) < self.RightLine.N:
            self.RightLine.fits.append(self.RightLine.current_fit)
        else:
            self.RightLine.fits.pop(0)
            self.RightLine.fits.append(self.RightLine.current_fit)

        return out_img

    @profile
    def detectCtn(self, binary_warped):
        """Continuous detection, based on previous detection"""
        # Update last fit
        self.LeftLine.last_fit = self.LeftLine.current_fit
        self.RightLine.last_fit = self.RightLine.current_fit
        # we know where the lines are in the previous frame
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = self.margin

        left_fit = self.LeftLine.current_fit
        right_fit = self.RightLine.current_fit

        left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                       left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                             left_fit[1] * nonzeroy + left_fit[
                                                                                 2] + margin)))

        right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                        right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                               right_fit[1] * nonzeroy + right_fit[
                                                                                   2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = self.ploty
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        dmax = np.max(np.absolute(left_fitx - right_fitx))
        dmin = np.min(np.absolute(left_fitx - right_fitx))

        if (dmax-dmin)/dmin > 0.4:# Not parallel
            # Reset detection
            if self.LeftLine.fail_num > 0:
                self.LeftLine.fail_num += 1
                if self.LeftLine.fail_num == self.LeftLine.MaxFail:
                    self.InitializedLD = False
        else:
            # reset failure number to zero
            self.LeftLine.fail_num = 0

        if dmax > 900 or dmin < 500:
            self.LeftLine.current_fit = np.mean(np.array(self.LeftLine.fits), axis=0)
            self.RightLine.current_fit = np.mean(np.array(self.RightLine.fits), axis=0)
        else:
            # Detection is true and update states
            if np.max(np.absolute(left_fitx - self.LeftLine.recent_xfitted))>70:
                self.LeftLine.current_fit = np.mean(np.array(self.LeftLine.fits), axis=0)  # self.LeftLine.last_fit
            else:
                self.LeftLine.current_fit = left_fit
                self.LeftLine.recent_xfitted = left_fitx

            if np.max(np.absolute(right_fitx - self.RightLine.recent_xfitted))>70:
                self.RightLine.current_fit = np.mean(np.array(self.RightLine.fits), axis=0)  # self.RightLine.last_fit
            else:
                self.RightLine.current_fit = right_fit
                self.RightLine.recent_xfitted = right_fitx

            # Pushback states
            if len(self.LeftLine.fits) < self.LeftLine.N:
                self.LeftLine.fits.append(self.LeftLine.current_fit)
            else:
                self.LeftLine.fits.pop(0)
                self.LeftLine.fits.append(self.LeftLine.current_fit)

            if len(self.RightLine.fits) < self.RightLine.N:
                self.RightLine.fits.append(self.RightLine.current_fit)
            else:
                self.RightLine.fits.pop(0)
                self.RightLine.fits.append(self.RightLine.current_fit)

        # Smooth the result
        self.LeftLine.current_fit = np.mean(np.array(self.LeftLine.fits), axis=0)#self.LeftLine.last_fit
        self.RightLine.current_fit = np.mean(np.array(self.RightLine.fits), axis=0)  # self.RightLine.last_fit

        self.LeftLine.diffs = left_fit - self.LeftLine.current_fit
        # self.LeftLine.current_fit = left_fit
        self.LeftLine.allx = leftx
        self.LeftLine.ally = lefty

        self.RightLine.diffs = right_fit - self.RightLine.current_fit
        # self.RightLine.current_fit = right_fit
        self.RightLine.allx = rightx
        self.RightLine.ally = righty

def test():
    # Read in the saved camera matrix and distortion coefficients
    dist_pickle = pickle.load( open( "camera_cal/wide_dist_pickle.p", "rb" ) )
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    M = dist_pickle["M"]
    Minv = dist_pickle["Minv"]

    a = Detector(mtx=mtx, dist=dist, M=M, Minv=Minv)
    a.setBinaryFun(flag=4)
    a.distance()
    a.LeftLine.test()
    a.debug = True

    img = mpimg.imread('test_images/straight_lines1.jpg')
    #img = mpimg.imread('./frames/1043.jpg')
    #img = mpimg.imread('./challenge_frames/0468.jpg')
    #img = img.astype(np.uint8)

    tmp = a.detect(img)
    plt.imshow(tmp)
    plt.show()

    img = mpimg.imread('./frames/0557.jpg')
    #img = mpimg.imread('test_images/straight_lines2.jpg')
    tmp = a.detect(img)

    plt.imshow(tmp)
    plt.show()
    print_prof_data()
    #a.plotFit()

def test2():
    kf = KalmanFilter(3)
    kf.printState()
    print('return ',kf.update(np.array([1.2,0.5,0.9])))
    print('return ', kf.update(np.array([1.2, 0.5, 0.9])+0.1))
    kf.printState()

def test3():
    print('rt ', np.eye(3))

def testKF():
    import pickle
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    dist_pickle = pickle.load(open("camera_cal/wide_dist_pickle.p", "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    M = dist_pickle["M"]
    Minv = dist_pickle["Minv"]

    detector = Detector(mtx=mtx, dist=dist, M=M, Minv=Minv, sx_thresh=(20, 100), s_thresh=(170, 255))
    q = [4, 4, 4]
    R = [1, 1, 1]
    detector.setKF_PR(q, R)
    detector.setMargin(60)
    detector.setBinaryFun(flag=5)
    detector.switchKF(False)

    # Import everything needed to edit/save/watch video clips
    from moviepy.editor import VideoFileClip
    from IPython.display import HTML

    white_output = 'output_images/project_output_v2_kf_tmp.mp4'
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    ##clip1 = VideoFileClip("./project_video.mp4").subclip(20,45)
    # clip1 = VideoFileClip("./project_video.mp4").subclip(0,1)
    clip1 = VideoFileClip("./challenge_video.mp4")  # .subclip(18,48)
    white_output = 'output_images/challenge_output_1.mp4'

    white_clip = clip1.fl_image(detector.detect) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)

#test3()
#test2()
#test()
#testKF()