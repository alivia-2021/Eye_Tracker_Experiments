import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # make picture gray
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]  # Cut the gray face frame out
        roi_color = img[y:y + h, x:x + w]  # Cut the face frame out

        # We’ll detect eyes in the same way. But on the face frame now, not the whole picture
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 2)

    cv2.imshow('img', img)
    k = cv2.waitKey(100) & 0xff
    if k == 27:   # wait for ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
