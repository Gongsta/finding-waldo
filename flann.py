#use: pip install -U opencv-contrib-python==3.4.2.16

import numpy as np
import cv2
import matplotlib.pyplot as plt

waldo = cv2.imread('glasses-cropped.jpg', 0)
waldo_background = cv2.imread('find/2.jpg', 0)

#Will return an error if cv2 not installed properly
sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(waldo, None)
kp2, des2 = sift.detectAndCompute(waldo_background, None)

#Using FLANN (Fast Library for Approximate Nearest Neighbors).  contains a collection of algorithms optimized for fast nearest neighbor search in large datasets and for high dimensional features
#FLANN PARAMETERS
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks= 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

#Store all the good matches as per Lowe's ratio test (view documentation)
good = []
for m, n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)



MIN_MATCH_COUNT = 4

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = waldo.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    waldo_background = cv2.polylines(waldo_background,[np.int32(dst)],True,255,10, cv2.LINE_AA)

else:
    print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(waldo,kp1,waldo_background,kp2,good,None,**draw_params)
plt.imshow(img3, 'gray')
plt.show()
plt.savefig('FLANN_results/trial6.png', dpi=1000)