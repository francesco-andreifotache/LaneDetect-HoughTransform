import cv2
import numpy as np

def canny(image, blur_ksize, canny_low, canny_high):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, blur_ksize, 0)
    canny_img = cv2.Canny(blur, canny_low, canny_high)
    return canny_img

def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]

    top_limit = int(height * 0.25)
    bottom_limit = int(height * 0.95)
    left_side = int(width * 0.05)
    right_side = int(width * 0.95)

    polygons = np.array([[
        (left_side, bottom_limit),
        (right_side, bottom_limit),
        (right_side, top_limit),
        (left_side, top_limit)
    ]])

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

cap = cv2.VideoCapture('./test-images/test9.mp4')
current_mode = 'easy'

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Videoclipul s-a terminat cu succes!")
        break

    frame = cv2.resize(frame, (500, 700))

    if current_mode == 'easy':
        blur_ksize = (5, 5)
        canny_low = 50
        canny_high = 150
        hough_thresh = 50
        hough_min_len = 50
        hough_max_gap = 10
        slope_thresh = 0.5
    else:
        blur_ksize = (15, 15)
        canny_low = 20
        canny_high = 80
        hough_thresh = 20
        hough_min_len = 10
        hough_max_gap = 120
        slope_thresh = 0.3

    canny_image = canny(frame, blur_ksize, canny_low, canny_high)
    cropped_image = region_of_interest(canny_image)

    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, hough_thresh, np.array([]), minLineLength=hough_min_len, maxLineGap=hough_max_gap)

    line_image = np.zeros_like(frame)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 == x2:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                continue

            slope = (y2 - y1) / (x2 - x1)

            if slope > slope_thresh or slope < -slope_thresh:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 3)

    combo_image = cv2.addWeighted(frame, 0.9, line_image, 1, 1)

    cv2.putText(combo_image, f"Mod: {current_mode.upper()}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Result', combo_image)
    cv2.imshow('Canny', canny_image)
    cv2.imshow("ROI", cropped_image)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('1'):
        current_mode = 'easy'
    elif key == ord('2'):
        current_mode = 'hard'
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()