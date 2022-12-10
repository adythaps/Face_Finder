import cv2
from random import randrange
#lode sum pre trained data
trained_face_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#chose an immage to detect a face

webcam = cv2.VideoCapture(0)

#iterate forever over frames
while True:
   # reed the current frame
   successful_frame_read, frame = webcam.read()
   #must convert to gray
   grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

   #detect face
   face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)



   for (x, y, w, h) in face_coordinates:
    cv2.rectangle(frame, (x, y), (x+w, y+w), (randrange(128,256), randrange(128,256), randrange(128,256)), 2)


   cv2.imshow('face finder', frame)
   key = cv2.waitKey(1)
   if key==81 or key==113:
       break 
webcam.release()










#img = cv2.imread('photo.jpg')
# for single photo/person(x, y, w, h) = face_coordinates[0]

#draw a rectangle
#cv2.rectangle(img, (40, 36), (40+82, 36+82), (0, 255, 0), 2)

#print(face_coordinates)

#cv2.imshow('face finder', frame)
#cv2.waitKey()