import cv2 as cv
import numpy as np
from tensorflow import keras

#Pre defined labels for the model
LABELS   = ['RED','BLUE']

# pre trained model to use
modelname = 'tf_model1.h5'
    
# load the model
trained_model =  keras.models.load_model(modelname)

# image prep for model input
IMG_HEIGHT = 480
IMG_WIDTH  = 270

def get_image_label(src_image):
    #img_arr = cv.imread(src_path)[...,::-1] #convert BGR to RGB format
    img_arr = cv.cvtColor(src_image, cv.COLOR_BGR2RGB)

    resized_img_arr = cv.resize(img_arr, (IMG_HEIGHT, IMG_WIDTH)) # Reshaping images to preferred size
        
    #flatten 3D array     
    flatten_image = resized_img_arr.reshape(-1, IMG_HEIGHT*IMG_WIDTH*3)
        
    # reshape/flatten input array for model input
    reshape_dim = (-1, IMG_HEIGHT, IMG_WIDTH, 3)
    flatten_image = flatten_image.reshape(reshape_dim)
      
        
    # predict image label index
    pred_dist = trained_model.predict(flatten_image,verbose = 0)
    pred_index = np.argmax(pred_dist[0])
        
    # predicted label 
    label = LABELS[pred_index]

    return label



vid = cv.VideoCapture(0)

frcnt = 0

while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    if not ret:
            break
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    label = get_image_label(frame)

    frcnt = frcnt + 1
    if frcnt == 60: 
        print(label)
        frcnt = 0
    
    #print(tags[0].tag_family)
    # Display the resulting frame
    cv.imshow('frame', frame)
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv.destroyAllWindows()