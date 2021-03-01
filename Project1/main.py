# importing libraries
import numpy as np
import cv2

# importing videos , image and webcam feed
imageTarget = cv2.imread("Resources/imgt.jpg")
video = cv2.VideoCapture("Resources/vid.mp4")
webCam = cv2.VideoCapture(0)

# getting height and width of Target image
height,width,z = imageTarget.shape

# finding keypoints and descriptor for Target image
orb = cv2.ORB_create(nfeatures=1000)
kp1, desc1 = orb.detectAndCompute(imageTarget,None)
# imageTarget = cv2.drawKeypoints(imageTarget,kp1,None)

detection = False
frameCounter = 0
_, vidImg = video.read()

while True:

    _,webCamImg = webCam.read()
    augmentedImage = webCamImg.copy()
    cam = webCamImg.copy()

    # finding keypoints and descriptor for Webcam image
    kp2, desc2 = orb.detectAndCompute(webCamImg,None)

    if detection==False:
        video.set(cv2.CAP_PROP_POS_FRAMES,0)
        frameCounter = 0
    else:
        if frameCounter == 900:
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0
        _, vidImg = video.read()
        vidImg = cv2.resize(vidImg, (width, height))

    # matching feature points of Target image and Webcam Feed using brute force matcher and knn method
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1,desc2,k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
    # print(len(good))

    if len(good) > 20:
        detection = True
        srcPoints = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dstPoints = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(srcPoints,dstPoints,cv2.RANSAC,5)
        # print(matrix)
        pts = np.float32([[0, 0], [0, height], [width, height], [width, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix)
        img2 = cv2.polylines(webCamImg, [np.int32(dst)], True, (255, 0, 255), 3)

        # wrapping the image
        imgWarp = cv2.warpPerspective(vidImg, matrix, (webCamImg.shape[1], webCamImg.shape[0]))

        # creating a new mask
        newMask = np.zeros((webCamImg.shape[0], webCamImg.shape[1]), np.uint8)
        cv2.fillPoly(newMask, [np.int32(dst)], (255, 255, 255))
        maskInv = cv2.bitwise_not(newMask)
        augmentedImage = cv2.bitwise_and(augmentedImage,augmentedImage,mask= maskInv)
        augmentedImage = cv2.bitwise_or(augmentedImage,imgWarp)
        frameCounter += 1
        # cv2.imshow("imgWarp", newMask)

    cv2.putText(cam, "WebCam feed", (0, 30), 1, cv2.FONT_HERSHEY_DUPLEX,(0, 150, 0), 2)
    cv2.putText(augmentedImage, "Augmented Video", (0, 30), 1, cv2.FONT_HERSHEY_DUPLEX, (0, 150, 0), 2)
    final = np.hstack((cam, augmentedImage))
    cv2.imshow("Final", final)
    # cv2.imshow("webCamImg", cam)
    # cv2.imshow("imgWarp", imgWarp)
    # cv2.imshow("augmentedImage", augmentedImage)

    print(frameCounter)  # 903
    if cv2.waitKey(1) & 0xff == 27:
        break

video.release()
webCam.release()
cv2.destroyAllWindows()


