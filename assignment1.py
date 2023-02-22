# # -*- coding: utf-8 -*-


import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim


############ 2.1 black/colored video (4 sec) ############

cap = cv.VideoCapture('new.mp4')

count = 0
while(1):
    success, img = cap.read()
    if success:
        count += 1
        if count %10 > 5:
            img = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
        cv.imshow('video',img)

        k = cv.waitKey(45) & 0xFF
        if k == ord('q'):
            cv.destroyAllWindows()
            break
    else:
        cv.destroyAllWindows()
        break


############ 2.2 smoothing by two means (8 sec) ############

# resize the image
def resize(src, scale_ratio):
    scale_ratio = scale_ratio
    width = int(src.shape[1] * scale_ratio)
    height = int(src.shape[0] * scale_ratio)
    return cv.resize(src, (width, height))

# two filters
def blur_show(img, method='G', wait=1000):
    cv.namedWindow('Image', 0)
    cv.imshow('Image', img)
    cv.waitKey(wait)
    
    for i in range(3,15,4):
        if method == 'G':
            bimg = cv.GaussianBlur(img,(i,i),6)
        else:
            bimg = cv.bilateralFilter(img,i,150,150)
        cv.imshow('Image', bimg)
        cv.waitKey(wait)
        cv.destroyAllWindows()

img = cv.imread('smpl.jpg')
img = resize(img, scale_ratio=0.4)
blur_show(img, method='G', wait=1000)
blur_show(img, method='B', wait=1000)



############ 2.3 grab leaves (8 sec) ############

def find_leaves(fill_holes=False):
    cap = cv.VideoCapture('new.mp4')
    kernel = np.ones((5,5),np.uint8)
    s = 30
    while(1):
        success, img = cap.read()
        if success:
            hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            lower_green = np.array([60-s,100,50])            
            upper_green = np.array([60+s,255,255])
            mask = cv.inRange(hsv, lower_green, upper_green)
            if fill_holes:
                mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
            cv.imshow('frame',img)
            cv.imshow('mask',mask)
            k = cv.waitKey(45) & 0xFF
            if k == 27:
                cv.destroyAllWindows()  
                break
        else:
            cv.destroyAllWindows()
            break 
        
find_leaves(fill_holes=False)   
find_leaves(fill_holes=True) 
    
    
    
############ 3.1 sobel edge detection (5 sec) ############
# %matplotlib qt5

img = cv.imread('frites.jpeg',0)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.axis('off'), ax2.axis('off'), ax3.axis('off')

frames = []
for i in range(3,16,4):
    sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=i)
    sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=i)
    im1 = ax1.imshow(img, cmap='gray', animated=True)
    im2 = ax2.imshow(sobelx, cmap='gray', animated=True)
    im3 = ax3.imshow(sobely, cmap='gray', animated=True)
    frames.append([im1, im2, im3])

ani = anim.ArtistAnimation(fig, frames, interval=1000, blit=True, repeat=False)



############ 3.2 Hough circular detection (10 sec) ############

def hough_circle_video(dist_deno = 16, dp = 1, p1 = 100, p2 = 30, minr = 1, maxr = 70):
    cap = cv.VideoCapture('circles.mp4')
    while(1):
        success, img = cap.read()
        if success:
            gry = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            gry = cv.medianBlur(gry, 5)
            row = gry.shape[0]
            crcl = cv.HoughCircles(gry, cv.HOUGH_GRADIENT, dp,  row / dist_deno, 
                                    param1=p1, param2=p2, minRadius=minr,
                                    maxRadius=maxr)
            if crcl is not None:
                crcl = np.uint16(np.around(crcl))
                for i in crcl[0,:]:
                    center = (i[0], i[1])
                    cv.circle(img, center, 1, (0,100,100), 3)
                    radius = i[2]
                    cv.circle(img, center, radius, (255,0,255), 3)
                    
            cv.imshow('detect circles',img)
            k = cv.waitKey(20) & 0xFF
            if k == 27:
                cv.destroyAllWindows()  
                break
        else:
            cv.destroyAllWindows()
            break 

hough_circle_video()
hough_circle_video(dist_deno = 32)
hough_circle_video(dp = 1.5)
hough_circle_video(p1 = 50)
hough_circle_video(p2 = 10)



############ 3.3 template matching (5 sec) ############

def track_rabbit(temp = 'rabbithead.png'):
    template = cv.imread(temp,0)
    w, h = template.shape[::-1]
    cap = cv.VideoCapture('rabbit.mp4')
    while(1):
        success, img = cap.read()
        if success:
            gry = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            res = cv.matchTemplate(gry,template,cv.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv.rectangle(img,top_left, bottom_right, (0,0,255), 10)
            cv.imshow('video',img)
            k = cv.waitKey(5) & 0xFF
            if k == 27:
                cv.destroyAllWindows()  
                break
        else:
            cv.destroyAllWindows()
            break 
      
track_rabbit()



def grey_scale():
    img = cv.imread('rabbitrun.png',0)
    template = cv.imread('rabbithead.png',0)
    res = cv.matchTemplate(img,template,cv.TM_CCOEFF_NORMED)
    
    plt.imshow(res,cmap = 'gray')
    cv.waitKey(3000)
    cv.destroyAllWindows()

grey_scale()



############ 4 carte blanche (20 sec) ############

# 4.1 tracking a different size cannot work with previous method 

track_rabbit(temp ='rabbitear_l.png')


# 4.2 change the size of the template 

template = cv.imread('rabbitear_l.png', 0)
scene = cv.imread('rabbitrun.png')

for i in np.linspace(0.1, 0.5, 10)[::-1]:
    temp_re = resize(template, scale_ratio=i)
    gry = cv.cvtColor(scene, cv.COLOR_BGR2GRAY)
    w, h = temp_re.shape[::-1]
    res = cv.matchTemplate(gry,temp_re,cv.TM_CCOEFF_NORMED)
    
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv.rectangle(scene,top_left, bottom_right, (0,0,255), 10)
    cv.imshow('rabbit_head',scene)
    k = cv.waitKey(1000) 
    cv.destroyAllWindows()
        

# 4.3 detecting image with rotation 

template = cv.imread('rabbitear_lr.png', 0)
scene = cv.imread('rabbitrun.png', 0)

sift = cv.SIFT_create()
keypoints1, des1= sift.detectAndCompute(template, None)
keypoints2, des2= sift.detectAndCompute(scene, None)

bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)
matches = bf.match(des1,des2)

matches = sorted(matches, key= lambda match : match.distance)
matched_imge = cv.drawMatches(template, keypoints1, scene, keypoints2, matches[:20], None)

cv.imshow("Matching Images", matched_imge)
cv.waitKey(3000)
cv.destroyAllWindows()


