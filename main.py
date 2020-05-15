import cv2

import numpy as np

from sklearn.linear_model import LinearRegression

# global constant variables for turning image into edge
KERNEL_SIZE = 5
THRESHOLD_MIN = 100
THRESHOLD_MAX = 150

# global constant variables for createing hough transform
RHO = 1             # the resolutio of paramter r in pixel, we are using 1
THETA = np.pi/180   # the value of theta in radians, here it is 1
THRESHOLD = 30      # the minimum number of intersections, 30
MIN_LINE_LENGTH = 2  # the min number of points that can form a line
# max line between two points to be considered in the same line
MAX_LINE_GAP = 150

# variables that are used in funciotns.

l_r_line = [[0, 0, 0, 0], [0, 0, 0, 0]]
l_r_sides = None

output = []

turn = None

# end of defning global variables


def get_canny_edge_detected(image=None):
    """the function receives an [image] as an argument

    The list of things that function performs
    1. checks if image is not None
    2. turns the color image to Gray Scale
    3. blurs the image to reduce the noise
    4. detects edges using canny edge detection
    """

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
    """Selects the region of interest and returns the mask of ROI

    The list of things funciton that does
    1. detects the height of given image
    2. generates the rectangular shape with type of ndarray
    3. creates the mask which is the same size as given frame
    4. takes the and of frame and the generated ROI - region of interest
    """

    height = image.shape[0]

    rectangle = np.array([
        [(345, height), (550, 550), (780, 550), (1100, height)]
    ], dtype=np.int32)

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, rectangle, 255)
    masked_image = cv2.bitwise_and(image, mask,)
    return masked_image


# this is a helper funciton for not reapating myself
def get_avg_slop_intercpt(x, y):
    """this function calculates and returns the avg slop of line using
    scikit linear model

    the arguments: x, y

    x - is the numpy array of all possible values for x
    y - is the numpy array of all possible values for y
    ---------------------------------------------------
    returns the slope and intersection.
    """
    new_model = LinearRegression().fit(x.reshape((-1, 1)), y)
    return [round(new_model.coef_[0], 3), new_model.intercept_]


def get_slope(x1, y1, x2, y2):
    """slope is used to seperate the lines to left and right
    """
    if round(x1 - x2, 2) == 0:
        return None
    else:
        return (y2 - y1) / (x2 - x1)


def get_seperated_lines(lines):
    """left and right side of lines are all in one result of hough transform

        lines - set of numpy arrays and send as an argument


        List of actions that this function does
        1. iterate over lines
        2. gets the slop of the ines
        3. stores the result into left or right side
        4. returns it
    """
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
    """This function returns the X and Y cordinates as separate.

        The list of things that this funcitons
        1. line_x and lines_y
        2. iterate over liens
        3. get x1, y1, x2, y2 from lines array
        4. appending x and y to lines
    """
    lines_x = []
    lines_y = []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        lines_x.extend([x1, x2])
        lines_y.extend([y1, y2])

    return [np.asarray(lines_x), np.asarray(lines_y)]


def make_cordinates(height, slp_intercept):
    """generates two set of cordinates

    1. gets the slope and intercept
    2. gets the heigh of image
    3. generates the y1 and y2
    4. generates the x1 and x2
    5. sends them in one array
    """
    slope, intercept = slp_intercept
    y1 = frame.shape[0]
    y2 = int(y1 * 0.72)
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)

    return [x1, y1, x2, y2]


def get_line_cordinates(height, l_r_sides):
    """"Function that returns the left and right side of cordinates

    1. This funciton depends on get_seperate_x_and_y,
        get_avg_slop_intercept and make_cordinates

    List of things that funtions does
    1. gets the separate x and y
    2. generates the slop and intercepts for x and y
    3. generates the cordinates for slop and intercept
    4. returns the cordinates that was generated

    it is used to draw regression line.

    """"
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
    """Display lines on blank image that were generated from get_line_cordinages

    The fuction does following actions
    1. generate blank image
    2. draw line for left side
    3. drwaw line for right side.
    """
    line_image = np.zeros_like(image)
    # if lines[0] is not None and lines[1] is not None:
    cv2.line(line_image, (left[0], left[1]),
             (left[2], left[3]), (255, 0, 0), 8)

    cv2.line(line_image, (right[0], right[1]),
             (right[2], right[3]), (255, 0, 0), 8)

    return line_image


# reading the video from file
cap = cv2.VideoCapture("video2.mp4")


# while the video is open we keep reading the frames.
while (cap.isOpened()):
    _, frame = cap.read()
    if frame is None:
        # if frame is none then we have to break the video playing
        break

    # detecting the edges from frame
    canny_edge = get_canny_edge_detected(frame)

    # getting the mask of edge detected image and ROI
    masked_image = get_region_of_interest(canny_edge)

    # generatign Hough transformaed lines
    lines = cv2.HoughLinesP(masked_image, RHO, THETA, THRESHOLD, np.array([]),
                            minLineLength=MIN_LINE_LENGTH, maxLineGap=MAX_LINE_GAP)

    # if any of lines is empty then we just move to the second frame
    if lines is not None:
        l_r_sides = get_seperated_lines(lines)

        # get_line_cordinates
        l_r_line = get_line_cordinates(frame.shape[0], l_r_sides)

    # displaying the generated left and right line onto blank image
    line_image = display_lines_on_blank(frame, l_r_line[0], l_r_line[1])

    # creating final frame that is merged with
    final_frame = cv2.addWeighted(frame, 0.8, line_image, 1, 0)

    difference = (115 - abs(l_r_line[0][2] - l_r_line[1][2]))
    if difference <= 5:
        turn = "left"
    elif difference >= 25:
        turn = "right"
    elif 8 < difference < 16:
        turn = "forward"
    else:
        turn = turn

    # writing text onto frame
    cv2.putText(final_frame, turn,
                (100, 100), 1, 2, (255, 255, 255))
    cv2.imshow("result", final_frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
