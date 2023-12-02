import cv2
import numpy as np


video_stream=  "http://192.168.1.96:8080/video" # CHANGE THIS

def preprocess(img):
    """
    this function take an image and prepares it to be used in finding contour and so on

    :param img: numpy array of the image
    :return image after being prepared and reduced to 1 channel
    """

    # convert to a single channel
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # blur to smooth out small noise edges
    blur = cv2.GaussianBlur(gray, (11, 11), 0)

    # detect edges
    canny = cv2.Canny(blur, 50, 150)

    # dilate and erode in order to eliminate noise and thicken the edges even further
    kernel = np.ones((5, 5))

    imgDilated = cv2.dilate(canny,kernel,iterations=2)
    morph = cv2.morphologyEx(imgDilated, cv2.MORPH_CLOSE, kernel)
    imgEroded = cv2.erode(morph,kernel,iterations=1)
    return imgEroded



def find_biggest_contour(img):
    """
    this function takes an image and finds the biggest contour in it

    :param img: numpy array of the image
    :return: biggest contour
    """

    # find contours
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # find the biggest contour

    # initialize varibales
    biggest = np.array([])
    max_area = 0

    # go through each contour found
    for cnt in contours:
        area = cv2.contourArea(cnt)

        # if the contour area is big enough
        if area > 5000:
            # approximate the contour aka reduce the number of point representing the contour also known as smoothing out
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            # if the approximation has more or less than 4 points this means we were not able to find a document of a rectangular shape
            # hence we will not proceed

            # even if it is a rectangle since we are looking for the biggest contour we compare it with the biggest one seen so far and
            # if it is bigger then we update
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area

    return biggest


def getWarpPerspective(img,width,height, biggest):
    """
    this function takes an image and finds the biggest contour in it

    :param img: numpy array of the image
    :return: biggest contour
    """


    biggest = biggest.reshape((4, 2))

    # get the corners of the biggest contour
    pts1 = np.float32(sorted(biggest,key=lambda x:(x[0],x[1])))

    # get the corners of the image
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    # get the transformation matrix
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    # apply the transformation matrix
    imgOutput = cv2.warpPerspective(img, matrix, (width, height))

    return imgOutput

def main():
    scaling_factor = 0.5

    capture = cv2.VideoCapture(video_stream)
    ret, frame = capture.read()
    if ret:
        width = int(frame.shape[1] * scaling_factor)
        height = int(frame.shape[0]* scaling_factor)
    
    while True:

        # read the video stream
        ret, frame = capture.read()

        frame = cv2.resize(frame,(0,0),fx= scaling_factor,fy = scaling_factor)

        contour_frame = frame.copy()

        if ret:
            imgThresh = preprocess(frame)
            cv2.imshow("Thresh", imgThresh)
            biggest = find_biggest_contour(imgThresh)
            if biggest.size != 0:
                cv2.drawContours(contour_frame, biggest, -1, (0, 255, 0), 20)
                warp = getWarpPerspective(frame,width,height,biggest)

                result = np.vstack((np.hstack((frame,cv2.cvtColor(imgThresh,cv2.COLOR_GRAY2BGR))),np.hstack((contour_frame,warp))))

            else:
                result = np.vstack((np.hstack((frame,cv2.cvtColor(imgThresh,cv2.COLOR_GRAY2BGR))),np.hstack((frame,frame))))

            

            cv2.imshow("Result", result)

        if cv2.waitKey(1) == ord('q'):
            break

    capture.release()





if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()