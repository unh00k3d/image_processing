import numpy as np
import cv2

def main():

    rows = 10
    columns = 10
    scaling_factor = 0.1

    image_count = rows * columns
    capture = cv2.VideoCapture(0)
    # get some data that will be used all the time
    ret , frame = capture.read()

    height = frame.shape[0]
    width = frame.shape[1]
    depth = frame.shape[2]
    # get the size of each image according to the number or rows and columns
    size_of_one_frame = ( int(height * scaling_factor) , int(width * scaling_factor))

    canvas_size = (rows * size_of_one_frame[0],columns * size_of_one_frame[1],depth)

    # create an empty canvas with the proper canvas size
    canvas = np.zeros(canvas_size,np.uint8)

    while True:
        ret , frame = capture.read()

        scaled = cv2.resize(frame,(0,0),fx=scaling_factor,fy=scaling_factor)

        # fill out the empty canvas with the specified number of rows and columns
        for row in range(rows):
            for column in range(columns):

                canvas[row * size_of_one_frame[0] : (row + 1) * size_of_one_frame[0] ,column * size_of_one_frame[1] : (column + 1) * size_of_one_frame[1]] =  rotate(scaled,(360/image_count) * (row * columns + (column + 1)))
            
        cv2.imshow('canvas',canvas)
            
        if cv2.waitKey(1) == ord('q'):
            break


    capture.release()
    cv2.destroyAllWindows()




def rotate(img,angle):
    rotMat = cv2.getRotationMatrix2D((img.shape[1]//2,img.shape[0]//2),angle,1)
    return cv2.warpAffine(img,rotMat,(img.shape[1],img.shape[0])) 


if __name__ == "__main__":
    main()