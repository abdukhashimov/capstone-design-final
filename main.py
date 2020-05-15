import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
import math

KERNEL_SIZE = 5
THRESHOLD_MIN = 100
THRESHOLD_MAX = 150

RHO = 1
THETA = np.pi/180
THRESHOLD = 30
MIN_LINE_LENGTH = 2
MAX_LINE_GAP = 150


def open_main_image(image_name=None):
    if image_name is None:
        raise ValueError("Please make sure you providede file name")
        sys.exit(0)
    try:
        image = cv2.imread(image_name)
    except Exception as e:
        print(e)
        sys.exit(0)

    return image


def get_canny_edge_detected(image=None):

    if image is None:
        raise ValueError("please provide image file!")
        sys.exit(0)

    # make it grey
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # noise reduce by gausian blur
    blured_image = cv2.GaussianBlur(grey, (KERNEL_SIZE, KERNEL_SIZE), 0)

    # cv2.imshow('blured_image', blured_image)
    # cv2.waitKey(5000)
    edges = cv2.Canny(blured_image, THRESHOLD_MIN, THRESHOLD_MAX)
    return edges


def get_region_of_interest(image):
    height = image.shape[0]
    rectangle = np.array([
        [(345, height), (550, 550), (780, 550), (1100, height)]
    ], dtype=np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, rectangle, 255)
    masked_image = cv2.bitwise_and(image, mask,)
    return masked_image


def get_avg_slop_intercpt(x, y):
    new_model = LinearRegression().fit(x.reshape((-1, 1)), y)
    return [round(new_model.coef_[0], 3), new_model.intercept_]


def get_slope(x1, y1, x2, y2):
    if round(x1 - x2, 2) == 0:
        return None
    else:
        return (y2 - y1) / (x2 - x1)


def get_seperated_lines(lines):
    left_side = []
    right_side = []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        slope = get_slope(x1, y1, x2, y2)
        if slope is None:
            return [left_side, right_side]
        if slope < 0:
            left_side.append(line[0])
        elif slope > 0:
            right_side.append(line[0])

    return [left_side, right_side]


def get_seperate_x_and_y(lines):
    lines_x = []
    lines_y = []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        lines_x.extend([x1, x2])
        lines_y.extend([y1, y2])

    return [np.asarray(lines_x), np.asarray(lines_y)]


def make_cordinates(height, slp_intercept):
    slope, intercept = slp_intercept
    y1 = frame.shape[0]
    y2 = int(y1 * 0.72)
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)

    return [x1, y1, x2, y2]


def get_line_cordinates(height, l_r_sides):
    left = l_r_sides[0]
    right = l_r_sides[1]
    output_left = []
    output_right = []
    if left is None or len(left) == 0:
        output_left = [0, 0, 0, 0]
    else:
        x_and_y_left = get_seperate_x_and_y(left)
        slp_intercept = get_avg_slop_intercpt(x_and_y_left[0], x_and_y_left[1])
        output_left = [0, 0, 0, 0] if slp_intercept[0] == 0 else make_cordinates(
            height, slp_intercept)

    if right is None or len(right) == 0:
        output_right = [0, 0, 0, 0]
    else:
        x_and_y_left = get_seperate_x_and_y(right)
        slp_intercept = get_avg_slop_intercpt(x_and_y_left[0], x_and_y_left[1])
        output_right = [0, 0, 0, 0] if slp_intercept[0] == 0 else make_cordinates(
            height, slp_intercept)

    return [output_left, output_right]


def display_lines_on_blank(image, left, right):
    line_image = np.zeros_like(image)
    # if lines[0] is not None and lines[1] is not None:
    cv2.line(line_image, (left[0], left[1]),
             (left[2], left[3]), (255, 0, 0), 8)

    cv2.line(line_image, (right[0], right[1]),
             (right[2], right[3]), (255, 0, 0), 8)

    return line_image


cap = cv2.VideoCapture("video2.mp4")

l_r_line = [[0, 0, 0, 0], [0, 0, 0, 0]]
l_r_sides = None
output = []
turn = None
out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc(
    'M', 'J', 'P', 'G'), 10, (1024, 768))
# out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1024, 768))
while (cap.isOpened()):
    _, frame = cap.read()
    if frame is None:
        break

    canny_edge = get_canny_edge_detected(frame)
    masked_image = get_region_of_interest(canny_edge)
    lines = cv2.HoughLinesP(masked_image, RHO, THETA, THRESHOLD, np.array([]),
                            minLineLength=MIN_LINE_LENGTH, maxLineGap=MAX_LINE_GAP)

    if lines is not None:
        l_r_sides = get_seperated_lines(lines)

        # get_line_cordinates
        l_r_line = get_line_cordinates(frame.shape[0], l_r_sides)
    line_image = display_lines_on_blank(frame, l_r_line[0], l_r_line[1])
    color_edges = np.dstack((canny_edge, canny_edge, canny_edge))

    final_frame = cv2.addWeighted(frame, 0.8, line_image, 1, 0)

    a = (115 - abs(l_r_line[0][2] - l_r_line[1][2]))
    if a <= 5:
        turn = "left"
    elif a >= 25:
        turn = "right"
    elif 8 < a < 16:
        turn = "forward"
    else:
        turn = turn

    cv2.putText(final_frame, turn,
                (100, 100), 1, 2, (255, 255, 255))
    cv2.imshow("result", final_frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
