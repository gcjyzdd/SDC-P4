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
        # Set lane line detection uninitialized
        self.InitializedLD = False

        self.undist = None
        # Define conversions in x and y from pixels space to meters
        self.ym_per_pix = 30 / 720  # meters per pixel in y dimension
        self.xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

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
        else:
            raise 'Invalid flag:'+str(flag)

    def performBinary(self, img):
        """Get the binary gradient output of a RGB image"""

        return self.Gradient.preprocess(img)

    def initDetection(self, binary_warped):
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
        margin = 50
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

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        self.ploty = ploty
        self.LeftLine.recent_xfitted = left_fitx
        self.RightLine.recent_xfitted = right_fitx
        return out_img

    def detect(self, img):
        """Detect lane lines on an image"""

        # preprocessing img
        self.undist = self._undistort(img)
        binary = self.performBinary(self.undist)
        binary_warped = self._warp(binary)

        if self.InitializedLD:
            # we know where the lines are in the previous frame
            nonzero = binary_warped.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            margin = 100

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

            self.LeftLine.diffs = left_fit-self.LeftLine.current_fit
            self.LeftLine.current_fit = left_fit
            self.LeftLine.allx = leftx
            self.LeftLine.ally = lefty

            self.RightLine.diffs = right_fit - self.RightLine.current_fit
            self.RightLine.current_fit = right_fit
            self.RightLine.allx = rightx
            self.RightLine.ally = righty

            # Generate x and y values for plotting
            ploty = self.ploty
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

            self.LeftLine.recent_xfitted = left_fitx
            self.RightLine.recent_xfitted = right_fitx
            return left_fit, right_fit, left_lane_inds, right_lane_inds
        else:
            # Reset the detection
            self.initDetection(binary_warped)
            pass

        #return self.visualizeInput(img)
        return self.visualizeDetection(binary_warped)

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
        left_fitx = self.LeftLine.recent_xfitted
        right_fitx = self.RightLine.recent_xfitted
        margin = 50
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

        result[ploty.astype(np.int32), left_fitx.astype(np.int32) ] = [255, 0, 255]
        result[ploty.astype(np.int32), right_fitx.astype(np.int32)] = [255, 0, 255]

        #plt.imshow(result)
        #plt.plot(left_fitx, ploty, color='yellow')
        #plt.plot(right_fitx, ploty, color='yellow')
        #plt.xlim(0, 1280)
        #plt.ylim(720, 0)
        #plt.show()
        return result

    def visualizeInput(self, img):
        """Plot the result on the input RGB image"""

        # Create an image to draw the lines on
        color_warp = np.zeros_like(img).astype(np.uint8)
        #color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        left_fitx = self.LeftLine.recent_xfitted
        right_fitx = self.RightLine.recent_xfitted
        ploty = self.ploty
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = self._invwarp(color_warp)#cv2.warpPerspective(color_warp, self.WarpMatrixInv, (img.shape[1], img.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(self.undist, 1, newwarp, 0.3, 0)

        #plt.imshow(result)
        #plt.show()
        return result

    def getCurvature(self):
        """Calculate curvature of two lines"""
        return 100,100

    def distance(self):
        print("The distance of two lines is .")
        return 1.0

    def _calculateCurvature(self, fit):
        pass

    def _gaussian_blur(img, kernel_size=5):
        """Applies a Gaussian Noise kernel"""
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    def _undistort(self, img):
        """Apply image undistortion"""
        return cv2.undistort(img, self.CameraMatrix, self.Distortion, None, self.CameraMatrix)

    def _warp(self, img):
        """Apply image warp transformation"""
        return cv2.warpPerspective(img, self.WarpMatrix, (img.shape[1], img.shape[0]))

    def _invwarp(self, img):
        """Apply image inverse warp transformation"""
        return cv2.warpPerspective(img, self.WarpMatrixInv, (img.shape[1], img.shape[0]))


# Read in the saved camera matrix and distortion coefficients
dist_pickle = pickle.load( open( "camera_cal/wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]
M = dist_pickle["M"]
Minv = dist_pickle["Minv"]

a = Detector(mtx=mtx, dist=dist, M=M, Minv=Minv)
a.distance()
a.LeftLine.test()

img = mpimg.imread('test_images/straight_lines1.jpg')
tmp = a.detect(img)


plt.imshow(tmp)
plt.show()

