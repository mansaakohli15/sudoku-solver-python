from image_reader import load_and_preprocess, find_biggest_contour
import cv2

img, thresh = load_and_preprocess(r"C:\Users\Asus\Downloads\sudoku.png")
grid = find_biggest_contour(thresh)

cv2.drawContours(img, [grid], -1, (0,255,0), 3)
cv2.imshow("Detected Sudoku", img)
cv2.waitKey(0)

