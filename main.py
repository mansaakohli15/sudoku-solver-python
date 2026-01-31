from image_reader import load_and_preprocess, find_biggest_contour, warp_perspective, split_cells
import cv2

img, thresh = load_and_preprocess(r"C:\Users\Asus\Downloads\sudoku.png")
grid = find_biggest_contour(thresh)

warped = warp_perspective(img, grid)
cells = split_cells(warped)

for i, cell in enumerate(cells):
    cv2.imshow(f"Cell {i+1}", cell)
    cv2.waitKey(100)
