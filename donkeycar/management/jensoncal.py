import cv2
assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np
import os
import sys
import glob
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

DIM=(160, 120)
K=np.array([[94.79596310883986, 0.0, 81.71060623709603], [0.0, 89.53176126621959, 61.57692671944012], [0.0, 0.0, 1.0]])
D=np.array([[-0.08030867117898016], [-0.04106184522378138], [0.04151960083911035], [-0.030190916823108125]])
M=np.array([[-1.3026235852665167, -3.499123877776032, 245.75446559156023], [-0.03176219298555294, -5.213807674195841, 254.91345254435038], [-0.000211747953236998, -0.02383023318793577, 1.0]])
def undistort(img_path):
    img = cv2.imread(img_path)
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img
    
def undistortImg(img):
    #img = cv2.imread(img_path)
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img
    
    
def undistort2(img_path, balance=0.0, dim2=None, dim3=None):
    img = cv2.imread(img_path)
    dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
    assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = dim1
    scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img

def undistort2Img(img, balance=0.0, dim2=None, dim3=None):
    dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
    assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = dim1
    scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img

def unwarp(img, src, dst, testing):
    h, w = img.shape[:2]
    # use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
    M = cv2.getPerspectiveTransform(src, dst)
    # use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)

    if testing:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        f.subplots_adjust(hspace=.2, wspace=.05)
        ax1.imshow(img)
        x = [src[0][0], src[2][0], src[3][0], src[1][0], src[0][0]]
        y = [src[0][1], src[2][1], src[3][1], src[1][1], src[0][1]]
        ax1.plot(x, y, color='red', alpha=0.4, linewidth=3, solid_capstyle='round', zorder=2)
        ax1.set_ylim([h, 0])
        ax1.set_xlim([0, w])
        ax1.set_title('Original Image', fontsize=30)
        ax2.imshow(cv2.flip(warped, 1))
        ax2.set_title('Unwarped Image', fontsize=30)
        plt.show()
    else:
        return warped, M

def lookAtFloorImg(img,maxWidth = 60, maxHeight = 30):
    uImg = undistortImg(img)
    warp = cv2.warpPerspective(uImg, M, (maxWidth+200,maxHeight+175))
    return warp
    
def lookAtFloorImg2(img,maxWidth = 300, maxHeight = 210,balance = 0.0):
    uImg = undistort2Img(img,balance = balance)
    warp = cv2.warpPerspective(uImg, M, (maxWidth,maxHeight))
    return warp
    
def lookAtFloor(img_path,maxWidth = 60, maxHeight = 30):
    uImg = undistort(img_path)
    warp = cv2.warpPerspective(uImg, M, (maxWidth+200,maxHeight+175))
    return warp
    
def getEdges(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 0, 200)
    return edged

if __name__ == "__main__":

    paths = glob.glob('/Users/edwardjackson/d_ej/data/tub_26_18-02-18/*.jpg')

    image = undistort2(paths[0],balance=0.5)
    orig=image.copy()
#




#ratio = image.shape[0] / 300.0
#orig = image.copy()
#image = imutils.resize(image, height = 300)
 
# convert the image to grayscale, blur it, and find edges
# in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)

#bothImgs = np.concatenate((gray, edged), axis=1)
#cv2.imshow('edge filter',bothImgs)


# find contours in the edged image, keep only the largest
# ones, and initialize our screen contour
    im2, contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(image, contours, -1, (0,255,0), 3)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
    paperCnt = None

# loop over our contours
    for c in contours:
        print('-----')
        sys.stdout.flush()# approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            print('found the paper')
            paperCnt = approx
            break
    cv2.drawContours(image, [paperCnt], 0, (0, 255, 0), 3)
    cv2.imshow("paper", image)


    pts = paperCnt.reshape(4, 2)
    rect = np.zeros((4, 2), dtype = "float32")
    rect = rect*2 
# the top-left point has the smallest sum whereas the
# bottom-right has the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
 
# compute the difference between the points -- the top-right
# will have the minumum difference and the bottom-left will
# have the maximum difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]




# now that we have our rectangle of points, let's compute
# the width of our new image
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
 
# ...and now for the height of our new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
 
# take the maximum of the width and height values to reach
# our final dimensions

#maxWidth = max((widthA),(widthB))
#ratio = 160/maxWidth
#rect = rect*ratio

#(tl, tr, br, bl) = rect
#widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
#widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
 
# ...and now for the height of our new image
#heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
#heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
    print('maxWidth')
    print(maxWidth)
    print('maxHeight')
    print(maxHeight)
# construct our destination points which will be used to
# map the screen to a top-down, "birds eye" view
    xOffset = 100
    yOffset = 150

    dst = np.array([
	    [0+xOffset, 0+yOffset],
	    [maxWidth - 1+xOffset, 0+yOffset],
	    [maxWidth - 1+xOffset, maxHeight - 1+yOffset],
	    [0+xOffset, maxHeight - 1+yOffset]], dtype = "float32")
 
# calculate the perspective transform matrix and warp
# the perspective to grab the screen
    M = cv2.getPerspectiveTransform(rect, dst)

    print("M=np.array(" + str(M.tolist()) + ")")

    warp = cv2.warpPerspective(orig, M, (maxWidth+200,maxHeight+175))
    raw = cv2.imread(paths[0])
    cv2.imshow("raw", raw)
    cv2.imshow("image", image)
    cv2.imshow("warp", warp)
    cv2.waitKey(0)