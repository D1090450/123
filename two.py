import numpy as np
import cv2

    # last_right_devia
def main(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    kernel_size = 3
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    low_threshold = 75
    high_threshold = 150
    img = cv2.Canny(img, low_threshold, high_threshold)

    left_deviation = 0
    for i in range(300,0,-1):
        if (img[100][i] == 255):
            left_deviation = i
            break

    right_deviation = 640
    for i in range(340,640):
        if (img[100][i] == 255):
            right_deviation = i
            break

    # print (left_deviation)
    # print (right_deviation)


    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.line(img, (0, 100), (300, 100), (0,255,0),3)
    cv2.line(img, (340, 100), (640, 100), (0,255,0),3)
    cv2.line(img, (320, 370), (320, 430), (255,255,255),3)
    cv2.line(img, (0, 70), (0, 130), (0,255,0),3)
    cv2.line(img, (300, 70), (300, 130), (0,255,0),3)
    cv2.line(img, (340, 70), (340, 130), (0,255,0),3)
    cv2.line(img, (640, 70), (640, 130), (0,255,0),3)
    cv2.line(img, (left_deviation, 80), (left_deviation, 120), (0,0,255),3)
    cv2.line(img, (right_deviation, 80), (right_deviation, 120), (0,0,255),3)
    turn = int((right_deviation + left_deviation)/2)
    cv2.line(img, (320, 400), (turn, 400), (255,0,0),3)
    # rho = 1
    # theta = np.pi/180
    # threshold = 10
    # min_line_len = 0
    # max_line_gap = 25
    # lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    # line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # for line in lines:
    #     # print (line)
    #     # linetext = "%d,%d"% (line[0][0], line[0][1]) 
    #     # cv2.putText(line_img, linetext, (line[0][0],line[0][1]), cv2.FONT_HERSHEY_PLAIN,
    #     #             1.0, (255,255,255), thickness = 1)
    #     # linetext = "%d,%d"% (line[0][2], line[0][3]) 
    #     # cv2.putText(line_img, linetext, (line[0][2],line[0][3]), cv2.FONT_HERSHEY_PLAIN,
    #     #             1.0, (0,255,255), thickness = 1)
    #     for x1,y1,x2,y2 in line:
    #         if (x2-x1)==0: continue
    #         if 10 > float(y2-y1)/(x2-x1) > 0:
    #             cv2.line(line_img, (x1, y1), (x2, y2), (255,0,0),5)
    #         elif 0 > float(y2-y1)/(x2-x1) > -10:
    #             cv2.line(line_img, (x1, y1), (x2, y2), (0,255,0),5)
    # line_img = cv2.cvtColor(line_img, cv2.COLOR_RGB2GRAY)



    # print (line_img.shape)
    # print (img.shape)
    # test_images_output = weighted_img(line_img, image, α=0.8, β=1., γ=0.)
    # test_images_output = cv2.addWeighted(img, 0.8, line_img, 1, 0)
    # print (test_images_output)
    return img
    # return line_img
    # return test_images_output


# cap = cv2.VideoCapture('video2.mp4')
cap = cv2.VideoCapture(0)
width = 480
height = 640
# print (width,height)
# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v' )
# out = cv2.VideoWriter(r'C:\\Users\\q2927\\Downloads\\yep\\output.mp4', fourcc, 20, (800,  600),True)
while True:
    ret, frame = cap.read()
    if ret:
        # frame = cv2.resize(frame,(800,600))
        # cv2.imshow('orgin',frame)
        frame = main(frame)
        # out.write(frame) 
        cv2.imshow('video',frame)
    else:
        break
    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
# out.release()
cv2.destroyAllWindows()

# img = cv2.imread('./test_images/test_image4.png')
# img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# low_threshold = 75
# high_threshold = 150
# img = cv2.Canny(img, low_threshold, high_threshold)
# print (img)
# def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         xy = "%d,%d" % (x, y)
#         print (xy)
#         cv2.circle(img, (x, y), 1, (255, 0, 0), thickness = -1)
#         cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
#                     1.0, (255,255,255), thickness = 1)
#         cv2.imshow("img", img)

# cv2.namedWindow("img")
# img = main(img)
# cv2.imshow('img',img)
# cv2.setMouseCallback("img", on_EVENT_LBUTTONDOWN)
# cv2.waitKey(0)

# while(True):
#     try:
#         cv2.waitKey(100)
#     except Exception:
#         cv2.destroyWindow("image")
#         break
        
# cv2.waitKey(0)
# cv2.destroyAllWindow()
#printing out some stats and plotting
# print('This image is:', type(image), 'with dimensions:', image.shape)