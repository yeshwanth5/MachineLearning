import cv2
import tensorflow as tf
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = tf.keras.models.load_model('C:\\Users\\yeshwanthr\\Documents\\Jupiter nb python files\\FERKaggleDataset\\FERModel.h5')
cap = cv2.VideoCapture(0)
while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read a frame.")
        break
    
    img=frame
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    #print(faces)
    try:
        (x,y,w,h) = faces[0]
        face = img[y:y+h,x:x+w]
        face_resize = cv2.resize(face,(48,48))
        face_new = np.expand_dims(face_resize,axis=0)
        #cv2.imshow('image',face_new[0])
        #arr = ['angry','disgust','fear','happy','neutral','sad','surprise']
        arr = ['happy', 'neutral', 'sad']
        emotion = arr[np.argmax(model.predict(face_new/255))]
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('image',frame)
    except:
        pass

    # Press 'q' to exit the loop
    key = cv2.waitKey(1)
    if key == 27:
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()