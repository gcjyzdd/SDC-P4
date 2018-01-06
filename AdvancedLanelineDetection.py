###

import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


# Read in the saved camera matrix and distortion coefficients
dist_pickle = pickle.load( open( "camera_cal/wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Load a test image
img = mpimg.imread('test_images/straight_lines1.jpg')

imshape = img.shape
vertices = np.array([[(50,imshape[0]),(500, 480), (800,480), (1200,imshape[0])]], dtype=np.int32)

# Undistortion
dst = cv2.undistort(img, mtx, dist, None, mtx)

# grayscale image
gray = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)

# sobel operation applied to x axis
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)

# Get absolute values
abs_sobelx = np.absolute(sobelx)

# Scale to (0, 255)
scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

# threshold
thresh_min = 20
thresh_max = 100
sxbinary = np.zeros_like(scaled_sobel)
sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
# plt.imshow(sxbinary, cmap='gray')


# Get HLS color image
hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)

# Get s-channel
s_channel = hls[:, :, 2]

# Threshold color channel
s_thresh_min = 170
s_thresh_max = 255
s_binary = np.zeros_like(s_channel)
s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1


# Combine grayscale and color gradient
combined_binary = np.zeros_like(sxbinary)
combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

combined_binary = region_of_interest(combined_binary, vertices)

kernel_size = 5
combined_binary = gaussian_blur(combined_binary, kernel_size)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(combined_binary, cmap='gray')
ax2.set_title('Combined Image', fontsize=30)

plt.show()

height = combined_binary.shape[0]
width = combined_binary.shape[1]

pts = np.array([[110,height],[500, 480], [780,480], [1200,height]], np.int32)
pts = pts.reshape((-1,1,2))
cv2.polylines(img,[pts],True,(255,0,0), thickness=2)
plt.imshow(img)
plt.show()
