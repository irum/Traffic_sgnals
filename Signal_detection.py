import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

### for location find contour ,rect on contour and then check for occlution


def img_detect(flags):

    image = cv2.imread(flags.data_path)
    im = np.copy(image)
    size = im.shape
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    ####finding contours

    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)
    canny = cv2.Canny(blurred, 120, 255, 1)

    ROI_No = 0
    cnts, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        ROI = im[y:y + h, x:x + w]
        #  cv2.imshow("im",ROI)
        cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 3)
        cv2.imwrite('ROI_{}.jpg'.format(ROI_No), ROI)
        ROI_No += 1

    # identify the range of colors in HSV
    red = np.uint8([[[255, 0, 0]]])
    hsvred = cv2.cvtColor(red, cv2.COLOR_BGR2HSV)
    print(hsvred)

    l_red = hsvred[0][0][0] - 10, 100, 100
    u_red = hsvred[0][0][0] + 10, 255, 255

    print(l_red)
    print(u_red)

    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    lower_red2 = np.array([30, 0, 20])  # 80, 0, 0])
    upper_red2 = np.array([200, 255, 255])  # [200, 255, 255]

    maskr = cv2.inRange(hsv, lower_red2, upper_red2)

    lower_green = np.array([40, 50, 50])
    upper_green = np.array([90, 255, 255])

    maskg = cv2.inRange(hsv, lower_green, upper_green)

    lower_yellow = np.array([60, 0, 50])
    upper_yellow = np.array([120, 255, 255])

    masky = cv2.inRange(hsv, lower_yellow, upper_yellow)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    lower_hue = np.array([160, 0, 50])
    upper_hue = np.array([180, 255, 255])

    mask_hue = cv2.inRange(hsv, lower_hue, upper_hue)

    ## TODO: use HoughCircles to detect circles
    # right now there are too many, large circles being detected
    # try changing the value of maxRadius, minRadius, and minDist
    r_circles = cv2.HoughCircles(maskr, cv2.HOUGH_GRADIENT, 1,
                                 minDist=120,
                                 param1=50,
                                 param2=5,
                                 minRadius=4,  # 4
                                 maxRadius=4)  # 12

    g_circles = cv2.HoughCircles(maskg, cv2.HOUGH_GRADIENT, 1, 60, param1=50, param2=10,
                                 minRadius=0, maxRadius=15)

    y_circles = cv2.HoughCircles(masky, cv2.HOUGH_GRADIENT, 1, 30, param1=50,
                                 param2=5,
                                 minRadius=3, maxRadius=15)

    # convert circles into expected type
    if r_circles is not None:
        circles = np.uint16(np.around(r_circles))

        # draw each one
        for i in circles[0, :]:
            # draw the outer circle

            cv2.circle(image, (i[0], i[1]), i[2], (0, 200, 0), 2)
            # draw the center of the circle
            cv2.circle(maskr, (i[0], i[1]), 2, (255, 255, 255), 3)
            print("Red signal detected")

    if g_circles is not None:
        circles = np.uint16(np.around(g_circles))

        # draw each one
        for i in circles[0, :]:
            # draw the outer circle
            # if areaContour == green_area:
            cv2.circle(image, (i[0], i[1]), i[2], (0, 200, 0), 2)
            # draw the center of the circle
            cv2.circle(maskg, (i[0], i[1]), 2, (255, 255, 255), 3)
            print("Green Signal detected")

    if y_circles is not None:
        circles = np.uint16(np.around(y_circles))

        # draw each one
        for i in circles[0, :]:
            # draw the outer circle

                cv2.circle(image, (i[0], i[1]), i[2], (0, 200, 0), 2)
                # draw the center of the circle
                cv2.circle(masky, (i[0], i[1]), 2, (255, 255, 255), 3)
                print("Yellow signal detected")

    # plt.imshow(image)

    #  print('Circles shape: ', circles.shape)

    cv2.imshow("Image", image)
    #  plt.imshow(mask, cmap='gray')
    # plt.show()

    cv2.waitKey(0)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/media/irum/DATA/job-tasks/cv-engineer-challenge-2/")

    flags = parser.parse_args()
    img_detect(flags)