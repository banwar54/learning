import cv2

img = cv2.imread("DEMO_1/input.jpg")
upscaled = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
cv2.imwrite("DEMO_1/output.jpg", upscaled)