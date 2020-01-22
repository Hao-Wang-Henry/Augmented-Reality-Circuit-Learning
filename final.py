#coding=utf-8
import cv2
import numpy as np
import scipy.signal as sp
from scipy import stats

def get_video():
    
    #address = "http://admin:admin@192.168.1.101:8081" ; locate_method = 4
    address = "video.mp4" ; locate_method = 1
    #address = 0; locate_method = 4

    res = 1  #low resolution, 400*300
    #res = 2  #median resolution, 640*480
    #res = 3  #high resolution, 960*720

    video =cv2.VideoCapture(address)
    cv2.namedWindow("breadboard")
    return video, locate_method, res

def kernel(kernel_size = 15):
    kernel = np.ones([kernel_size,kernel_size])
    kernel/=kernel_size**2
    return kernel

def drawlines(img, rho, theta):
    a   =  np.cos(theta)
    b   =  np.sin(theta)
    x0  =  a*rho
    y0  =  b*rho
    x1  =  int(x0  +  2000*(-b))
    y1  =  int(y0  +  2000*(a))
    x2  =  int(x0  -  2000*(-b))
    y2  =  int(y0  -  2000*(a))
    cv2.line(img,(x1,y1),(x2,y2),(255,255,255),2)

def findcorners(img, rho1, rho2, theta, par2):

    coordinate_tem = np.zeros([4,2])
    coordinate = np.zeros([4,2])
    
    rho4 = 0.25*rho1 + 0.75*rho2
    rho3 = 0.25*rho2 + 0.75*rho1

    a = np.cos(theta)
    b = np.sin(theta)
    x03 = a*rho3
    y03 = b*rho3
    x04 = a*rho4
    y04 = b*rho4

    x10 = int(par2/4)
    x20 = int(3*par2/4)
    y103 = int(y03 + (x10-x03)*(a)/(-b))
    y104 = int(y04 + (x10-x04)*(a)/(-b))
    y203 = int(y03 + (x20-x03)*(a)/(-b))
    y204 = int(y04 + (x20-x04)*(a)/(-b))

    x1 = x10
    y13 = int(y03 + (x1-x03)*(a)/(-b))
    y14 = int(y04 + (x1-x04)*(a)/(-b))

    x2 = x20
    y23 = int(y03 + (x2-x03)*(a)/(-b))
    y24 = int(y04 + (x2-x04)*(a)/(-b))

    #left up
    while img[y13][x1]==255.:
        x1-=10
        y13 = int(y03 + (x1-x03)*(a)/(-b))
    x1-=10
    if img[y13][x1]==0:
        x1+=11
        y13 = int(y03 + (x1-x03)*(a)/(-b))
        while img[y13][x1]==0:
            x1+=1
            y13 = int(y03 + (x1-x03)*(a)/(-b))
    else:
        print("error in left up")
    coordinate_tem[0][0] = x1
    coordinate_tem[0][1] = y13

    #left down
    x1 = x10
    while img[y14][x1]==255:
        x1-=10
        y14 = int(y04 + (x1-x04)*(a)/(-b))
    x1-=10
    if img[y14][x1]==0:
        x1+=11
        y14 = int(y04 + (x1-x04)*(a)/(-b))
        while img[y14][x1]==0:
            x1+=1
            y14 = int(y04 + (x1-x04)*(a)/(-b))
    else:
        print("error in left down")
    coordinate_tem[1][0] = x1
    coordinate_tem[1][1] = y14

    #right up
    while img[y23][x2]==255:
        x2+=10
        y23 = int(y03 + (x2-x03)*(a)/(-b))
    x2+=10
    if img[y23][x2]==0:
        x2-=11
        y23 = int(y03 + (x2-x03)*(a)/(-b))
        while img[y23][x2]==0:
            x2-=1
            y23 = int(y03 + (x2-x03)*(a)/(-b))
    else:
        print("error in right up")
    coordinate_tem[3][0] = x2 
    coordinate_tem[3][1] = y23

    #right down
    x2 = x20
    while img[y24][x2]==255:
        x2+=10
        y24 = int(y04 + (x2-x04)*(a)/(-b))
    x2+=10
    if img[y24][x2]==0:
        x2-=11
        y24 = int(y04 + (x2-x04)*(a)/(-b))
        while img[y24][x2]==0:
            x2-=1
            y24 = int(y04 + (x2-x04)*(a)/(-b))
    else:
        print("error in right down")
    coordinate_tem[2][0] = x2 
    coordinate_tem[2][1] = y24

    coordinate[0][0] = 1.5*coordinate_tem[0][0] - 0.5*coordinate_tem[1][0]
    coordinate[0][1] = 1.5*coordinate_tem[0][1] - 0.5*coordinate_tem[1][1]
    coordinate[1][0] = 1.5*coordinate_tem[1][0] - 0.5*coordinate_tem[0][0]
    coordinate[1][1] = 1.5*coordinate_tem[1][1] - 0.5*coordinate_tem[0][1]
    coordinate[3][0] = 1.5*coordinate_tem[3][0] - 0.5*coordinate_tem[2][0]
    coordinate[3][1] = 1.5*coordinate_tem[3][1] - 0.5*coordinate_tem[2][1]
    coordinate[2][0] = 1.5*coordinate_tem[2][0] - 0.5*coordinate_tem[3][0]
    coordinate[2][1] = 1.5*coordinate_tem[2][1] - 0.5*coordinate_tem[3][1]
    
    return coordinate
        
def calibrate(image, res):
    
    print("auto-calibration")
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.Canny(img, 100, 100)
    G7 = kernel(7)
    G9 = kernel(9)
    if res == 1:
        img= sp.convolve2d(img, G7, mode='same', boundary='symm')
        img= sp.convolve2d(img, G7, mode='same', boundary='symm')
        img  = np.where(img > 40, 255., 0.)
        (par1, par2, par3) = (4, 400, 300)
    elif res == 2:
        img= sp.convolve2d(img, G7, mode='same', boundary='symm')
        img= sp.convolve2d(img, G7, mode='same', boundary='symm')
        img= sp.convolve2d(img, G7, mode='same', boundary='symm')
        img  = np.where(img > 30, 255., 0.)
        (par1, par2, par3) = (8, 640, 480)
    elif res == 3:
        img= sp.convolve2d(img, G9, mode='same', boundary='symm')
        img= sp.convolve2d(img, G9, mode='same', boundary='symm')
        img= sp.convolve2d(img, G9, mode='same', boundary='symm')
        img= sp.convolve2d(img, G9, mode='same', boundary='symm')
        img  = np.where(img > 20, 255., 0.)
        (par1, par2, par3) = (9, 960, 720)
    tem1 = img
    
    img = np.float32(img)
    dst = cv2.cornerHarris(img,2,3,0.01)
    dst = cv2.dilate(dst,None)
    img = np.where(dst>0.001*dst.max(), 255, 0)
    img = img.astype('uint8')
    img = cv2.Canny(img, 230, 250)
    
    threshold = 250
    count = [0]
    while count[0]<par1:
        lines  =  cv2.HoughLines(img,1,np.pi/180,threshold)
        if not lines is None:
            theta = lines[:,0,1]
            mode, count = stats.mode(theta)
        threshold-=10
    theta = mode[0]
    lines = lines[lines[:,0,1]==theta]
    lines = lines[lines[:,0,0].argsort()]

    #draw long edges
    #drawlines(img, lines[0,0,0], lines[0,0,1])
    #drawlines(img, lines[len(lines)-1,0,0], lines[len(lines)-1,0,1])

    #find corners
    loc = findcorners(tem1, lines[0,0,0], lines[len(lines)-1,0,0], theta, par2)

    par4 = min(loc[0][0],loc[0][1],(loc[1][1]-loc[0][1])/2,50)
    bbox1 = (loc[0][0]-par4, loc[0][1]-par4, 2*par4, 2*par4)
    par4 = min(loc[1][0],par3-loc[1][1],(loc[1][1]-loc[0][1])/2,50)
    bbox2 = (loc[1][0]-par4, loc[1][1]-par4, 2*par4, 2*par4)
    par4 = min(par2-loc[2][0],par3-loc[2][1],(loc[2][1]-loc[3][1])/2,50)
    bbox3 = (loc[2][0]-par4, loc[2][1]-par4, 2*par4, 2*par4)
    par4 = min(par2-loc[3][0],loc[3][1],(loc[2][1]-loc[3][1])/2,50)
    bbox4 = (loc[3][0]-par4, loc[3][1]-par4, 2*par4, 2*par4)
    
    tracker = cv2.MultiTracker_create()
    tracker.add(cv2.TrackerKCF_create(), image, bbox1)
    tracker.add(cv2.TrackerKCF_create(), image, bbox2)
    tracker.add(cv2.TrackerKCF_create(), image, bbox3)
    tracker.add(cv2.TrackerKCF_create(), image, bbox4)
    boxes = [bbox1, bbox2, bbox3, bbox4]

    return tracker, boxes
    
def track(image, tracker):
    
    timer = cv2.getTickCount()
    ok, boxes = tracker.update(image)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    
    cv2.putText(image, "FPS : " + str(int(fps)), (10,60), \
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
    cv2.putText(image, "Tracking, KCF tracker", (10,30), \
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
    return boxes
    
def manual_calibrate(image):
    cv2.putText(image, "Manually calibrating", (10,30), \
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)

    bbox1 = cv2.selectROI('breadboard', image)
    bbox2 = cv2.selectROI('breadboard', image)
    bbox3 = cv2.selectROI('breadboard', image)
    bbox4 = cv2.selectROI('breadboard', image)

    tracker = cv2.MultiTracker_create()
    tracker.add(cv2.TrackerKCF_create(), image, bbox1)
    tracker.add(cv2.TrackerKCF_create(), image, bbox2)
    tracker.add(cv2.TrackerKCF_create(), image, bbox3)
    tracker.add(cv2.TrackerKCF_create(), image, bbox4)
    boxes = [bbox1, bbox2, bbox3, bbox4]

    return tracker, boxes

def display(image, boxes):
    
    for box in boxes:
        p1 = int(box[0] + 0.5*box[2])
        p2 = int(box[1] + 0.5*box[3])
        cv2.circle(image, (p1, p2), 5, (0,0,255), -1)
    image=cv2.resize(image,(640,480),interpolation=cv2.INTER_CUBIC)
    cv2.imshow("breadboard",image)

def AR():
    a = 1

if __name__ == '__main__':

    video, locate_method, res = get_video()
    (num, operation, boxes) = (0, 0, [])
    time = cv2.getTickCount()

    #if True:
    while True:
        ok, frame = video.read()
        if not ok:
            print('No video')
            break
            
        #choose resolution from low, median, high
        if res == 1:
            frame=cv2.resize(frame,(400,300),interpolation=cv2.INTER_CUBIC)
        elif res == 2:
            frame=cv2.resize(frame,(640,480),interpolation=cv2.INTER_CUBIC)
        elif res == 3:
            frame=cv2.resize(frame,(960,720),interpolation=cv2.INTER_CUBIC)

        #choose locating method from aotu, maunal, and track                    
        if locate_method == 1:
            #auto-calibrate
            try:
                tracker, boxes = calibrate(frame, res)
                locate_method = 2
            except:
                print("calibrate fail, re-calibrating")
            time = cv2.getTickCount()
        elif locate_method == 2:
            #track
            boxes = track(frame, tracker)
        elif locate_method == 3:
            #manually calibrate
            tracker, boxes = manual_calibrate(frame)
            locate_method = 2
            time = cv2.getTickCount()

        #choose operation from display the dots and AR implementation
        if operation == 0:
            #display dots on the corner
            display(frame, boxes)  
        elif operation == 1:
            #place components on board
            AR()

        #auto calibrate for every 5 seconds        
        if (cv2.getTickCount()-time)//cv2.getTickFrequency() >= 5:
            if locate_method == 2:
                time = cv2.getTickCount()
                locate_method = 1

        #deal with keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            #press esc to escape
            print("esc break...")
            break
        elif key == ord(' '):
            print("Pause")
            key = cv2.waitKey(0) & 0xFF
            if key == ord('s'):
                print("save current frame")
                num = num+1
                filename = "frames_%s.jpg" % num
                cv2.imwrite(filename, frame)
            if key == ord('m'):
                print("manual calibration")
                locate_method = 3
            if key == ord('c'):
                locate_method = 1
        elif key == ord('m'):
            print("manual calibration")
            locate_method = 3
        elif key == ord('c'):
            locate_method = 1


    video.release()
    cv2.destroyWindow("breadboard")

