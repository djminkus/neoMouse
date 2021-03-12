import cv2
import dlib
import numpy as np

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# Detect faces in an image:
img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grayscale detector = dlib.get_frontal_face_detector()
detector = dlib.get_frontal_face_detector()
rects = detector(gray, 1)                     # rects contains all the faces detected

# Draw rectangle around detected face:
# print(rects[0])
# cv2.rectangle(img, rects[0][0], rects[0][1])

# Getting issue because each entry of rects is a dlib rectangle object
# I need to do this in reverse: https://github.com/davisking/dlib/issues/1359
# Skipping for now


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


predictor = dlib.shape_predictor('shape_68.dat')
for (i, rect) in enumerate(rects):
    shape = predictor(gray, rect)
    shape = shape_to_np(shape)
    for (x, y) in shape:
        cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

cv2.imwrite('output.jpg', img)

cap = cv2.VideoCapture(0)
while(True):
    ret, img = cap.read()
    cv2.imshow("Output", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): # escape when q is pressed
        break


def eye_on_mask(mask, side):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask


left = [36, 37, 38, 39, 40, 41]              # keypoint indices for left eye
right = [42, 43, 44, 45, 46, 47]             # keypoint indices for right eye
mask = np.zeros(img.shape[:2], dtype=np.uint8)
mask = eye_on_mask(mask, left)
mask = eye_on_mask(mask, right)