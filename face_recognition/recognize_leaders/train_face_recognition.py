import cv2
import numpy as np
import os

def load_data(main_dir):
    """
    :param main_dir: base directory for the data
    :return: two numpy arrays one for the images and one for the labels
    """
    images = []
    labels = []
    directories = os.listdir(main_dir)
    if '.DS_Store' in directories: directories.remove('.DS_Store')
    print(f"labels : {directories}\n\n")
    for i,dir in enumerate(directories):
        for file in os.listdir(os.path.join(main_dir, dir)):
            image = cv2.imread(os.path.join(main_dir, dir, file))
            images.append(image)
            labels.append(i)
    return np.array(images,dtype='object'), np.array(labels)


def convert_to_grayscale(images):
    """
    :param images: numpy array of images
    :return: numpy array of grayscale images
    """

    
    gray_images = []
    for image in images:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_images.append(gray_image)

    
    return np.array(gray_images,dtype = 'object')


def pick_recognized_face_images(images , labels,classifier_path):
    """
    :param images: numpy array of images (images should be in grayscale)
    :param labels: numpy array of labels
    :return: numpy array of images with only the recognized face
    """
    cascade_classifier = cv2.CascadeClassifier(classifier_path)
    recognized_face_images = []
    recognized_labels  = []
    for i, image in enumerate(images):
        faces = cascade_classifier.detectMultiScale(image, 1.3, 5)
        if len(faces) == 1:
            for (x,y,w,h) in faces:
                recognized_face_images.append(image[y:y+h, x:x+w]) # take the recognized face only
                recognized_labels.append(labels[i])
    return np.array(recognized_face_images,dtype='object'), np.array(recognized_labels)
    

def train_model(images, labels):
    """
    :param images: numpy array of images (these are cropped face images)
    :param labels: numpy array of labels
    :return: trained model
    """
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(images, labels)    
    return model



if __name__ == '__main__':
    main_dir = "assets/leaders"
    print(f"loading data from {main_dir}...\n")
    images, labels = load_data(main_dir)
    print(f"\nloading data from {main_dir} done ✅\n")
    # gray scale images
    print(f"converting image to grayscale...\n")
    gray_images = convert_to_grayscale(images)
    print(f"converting image to grayscale done ✅\n")
    # pick only the recognized faces
    print(f"picking up faces that can be recognized...\n")
    gray_face_images, face_labels = pick_recognized_face_images(gray_images, labels, "assets/cascades/haarcascade_frontalface_default.xml")


    # shuffle the data to avoid bias toward a particular class
    shuffler = np.arange(gray_face_images.shape[0])
    np.random.shuffle(shuffler)
    gray_face_images = gray_face_images[shuffler]
    face_labels = face_labels[shuffler]
    
    unique_labels, unique_counts = np.unique(face_labels,return_counts=True)
    for label,count in zip(unique_labels,unique_counts):
        print(f"label {label} has {count} images recognized")
    print(f"\npicking up faces that can be recognized done ✅ \n")

    print(f"model training starting now...")
    model = train_model(gray_face_images, face_labels)
    print(f"model training done ✅\n")
    # save the training data and trained model to be used later
    model.save("assets/trained_model.yml")
    np.save("assets/face_recognition_gray_images.npy", gray_images)
    np.save("assets/face_images.npy", gray_face_images)
    np.save("assets/face_labels.npy", face_labels)



