import cv2 as cv
import numpy as np

img_dir = '/Users/aadipatangi/Desktop/Python Projects/ConeDetection'
tem_path = img_dir+'/template1.jpeg'
template = cv.imread(tem_path, 0)
tempsized = cv.resize(template, (212, 300)) 

def get_image_label(src_image):

    #img_gray = cv.cvtColor(src_image, cv.COLOR_BGR2GRAY)

    # Perform match operations.
    res = cv.matchTemplate(src_image, tempsized, cv.TM_CCOEFF_NORMED)
 
    # Specify a threshold
    threshold = 0.75
    # Store the coordinates of matched area in a numpy array
    loc = np.where(res >= threshold)

    detected = False

    for pt in zip(*loc[::-1]):
        detected = True
        break
        #cv.rectangle(src_image, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 1)

    return detected



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
    if frcnt == 2: 
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