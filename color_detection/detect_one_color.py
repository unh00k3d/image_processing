import numpy as np
import cv2


def get_colors():

    # get the new values
    h_min = cv2.getTrackbarPos('hue min','Control')
    h_max = cv2.getTrackbarPos('hue max','Control')
    s_min = cv2.getTrackbarPos('sat min','Control')
    s_max = cv2.getTrackbarPos('sat max','Control')
    v_min = cv2.getTrackbarPos('vue min','Control')
    v_max = cv2.getTrackbarPos('vue max','Control')
    return h_min,h_max,s_min,s_max,v_min,v_max


def init():
    """
    create the trackbars and so on to start detecting colors
    """
    # create a windows for controling the color that we want to detect
    cv2.namedWindow('Control')
    # each time any parameter is updated the show_colors function will be called and it will update the image that show the color gradient accordingly
    cv2.createTrackbar('hue min','Control',0,179,lambda x : None)
    cv2.createTrackbar('hue max','Control',255,179,lambda x:None)
    cv2.createTrackbar('sat min','Control',0,255,lambda x:None)
    cv2.createTrackbar('sat max','Control',255,255,lambda x:None)
    cv2.createTrackbar('vue min','Control',0,255,lambda x:None)
    cv2.createTrackbar('vue max','Control',255,255,lambda x:None)




def main():
    """
    our main code will be here
    this funcition will be responsible for 
    1. displaying the original image
    2. displaying the mask
    3. displaying the masked image (i.e. apply the mask to the image)
    """


    # define how much will we shrink each image
    scaling_factor = 0.5


    window_size = (480,640)


    # get the size of each image according to the number or rows and columns
    size_of_one_image = ( int(window_size[0] * scaling_factor) , int(window_size[1] * scaling_factor))

    # NOTE : if this is an image then we will only update the images when we get a function call that the parameters have been updated
    # however if this is a video then we will update the images every frame 




    # lets start with the image code
    # load the image
    img = cv2.imread('assets/gray_pen.jpg')
    # convert the image to HSV
    hsv_image = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    img_resized = cv2.resize(img,size_of_one_image[::-1])
    hsv_image_resized = cv2.resize(hsv_image,size_of_one_image[::-1])

    # for more responsiveness i will concatenate the original and hsv version since they will no change
    concatenated_original_hsv_image  = np.hstack((img_resized,hsv_image_resized))

    while True:
        # get the new values
        h_min,h_max,s_min,s_max,l_min,l_max = get_colors()

        # create the mask
        mask = cv2.inRange(hsv_image_resized,np.array([h_min,s_min,l_min]),np.array([h_max,s_max,l_max]))



        # apply the mask to the image
        masked_image = cv2.bitwise_and(img_resized,img_resized,mask=mask)

        mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)

        final = np.vstack((concatenated_original_hsv_image,np.hstack((masked_image,mask))))
        

        cv2.imshow('images',final)

        # NOTE : if this is an then we will only update the images when we get a function call that the parameters have been updated
        # however if this is a video then we will update the images every frame 
        # we will use the waitKey function to check if the user has pressed the q key to quit the program


        if cv2.waitKey(1) == ord('q'):
            break

    







if __name__ == "__main__":
    init()
    main()
    cv2.destroyAllWindows()