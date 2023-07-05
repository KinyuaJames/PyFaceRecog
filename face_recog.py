import cv2

# reading the cascade xml
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

# reading the image
img = cv2.imread('./test2.jpg')

#converting the image to greyscale 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# getting the rectangle points for the image
for(x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h),(225,0,0), 2)

# output the image 
cv2.imshow('img', img)
cv2.waitKey()

cv2.imwrite('face_detected', img)
