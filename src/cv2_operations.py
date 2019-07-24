import sys
import cv2
import numpy as np

def camera_capture():
    cap = cv2.VideoCapture(0)

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Display the resulting frame
        cv2.imshow('CS512_HW2', frame)

        if cv2.waitKey(1) & 0xFF != 255:
            cap.release()
            return frame

def cv2_smooth(x): #function called whenever the smoothing s trackbar value change, smooth the image with opencv library
    global img
    if (len(img.shape) != 3):
        print("not enough channels to convert to grey")
        img_smooth = img.copy()
    else:
        img_smooth = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x = 2*x + 1
    img_smooth = cv2.blur(img_smooth, (x, x))
    if(cv2_smooth_mode): #we show the updated image only in the corresponding mode
        cv2.imshow('CS512_HW2', img_smooth)

def my_smooth(x): #function called whenever the smoothing S trackbar value change, smooth the image with my operations
    global img
    global img_mysmooth
    if (my_smooth_mode):
        if (len(img.shape) != 3):
            print("not enough channels to convert to grey")
            img_smooth = img.copy()
        else:
            img_smooth = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if x == 0 :
            print("too small to apply a visible smoothing")
            return

        #the 3 user input correspond to these sizes:
        #  1 = 3x3 matrice
        #  2 = 5x5 matrice
        #  3 = 7x7 matrice
        x = 3 if x==1 else (5 if x==2 else 7)

        img_smoothed = img_smooth.copy()

        #we choose to ignore the pixels at the boundaries

        for ind_x in range(x+2, img_smooth.shape[1]-x-2):
            for ind_y in range(x+2, img_smooth.shape[0]-x-2):
                mat_temp = img_smooth[ind_y-(x-1)/2:ind_y+(x-1)/2+1, ind_x-(x-1)/2:ind_x+(x-1)/2+1]
                img_smoothed[ind_y][ind_x] = np.sum(mat_temp/(x**2))

        #Only for this one, we pu the result in a global varaible, so we don't have to do the computation again
        img_mysmooth = img_smoothed.copy()

        cv2.imshow('CS512_HW2', img_smoothed)

def cv2_rotate(x): #function called whenever the rotation trackbar value change, rotate the image
    global img
    if (len(img.shape) != 3):
        print("not enough channels to convert to grey")
        img_rot = img.copy()
    else:
        img_rot = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cols = img_rot.shape[0]
    rows = img_rot.shape[1]
    center = (rows/2, cols/2)
    mat_rot = cv2.getRotationMatrix2D(center, x, 1.0)
    img_rotated = cv2.warpAffine(img_rot, mat_rot, (rows, cols))
    if (cv2_rotation_mode): #we show the updated image only in the corresponding mode
        cv2.imshow('CS512_HW2', img_rotated)

def gradient_field(x):
    global img
    global img_gradVect
    if (len(img.shape) != 3):
        print("not enough channels to convert to grey")
        img_grad = img.copy()
    else:
        img_grad = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if x < 25:
        x = 25

    for ind_x in xrange(0, img_grad.shape[1], x):
        min_x = ind_x
        max_x = ind_x + x
        for ind_y in xrange(0, img_grad.shape[0], x):
            min_y = ind_y
            max_y = ind_y + x

            #draw the square for the matrices
            cv2.line(img_grad, (min_x, min_y), (min_x, max_y), (0, 125, 125), thickness=1, lineType=8, shift=0)
            cv2.line(img_grad, (min_x, max_y), (max_x, max_y), (0, 125, 125), thickness=1, lineType=8, shift=0)
            cv2.line(img_grad, (max_x, max_y), (max_x, min_y), (0, 125, 125), thickness=1, lineType=8, shift=0)
            cv2.line(img_grad, (max_x, min_y), (min_x, min_y), (0, 125, 125), thickness=1, lineType=8, shift=0)

            #get the corresponding matrice
            mat_temp = img_grad[min_y:max_y, min_x:max_x]

            #calculate the center of the matrice
            center = (min_x + x / 2, min_y + x / 2)

            #make sure the matrice is complete
            if(mat_temp.shape[0] != 0 and mat_temp.shape[1] != 0):

                #calculate of the gradient in y axis
                grad_y = mat_temp.copy()
                grad_y = cv2.Sobel(grad_y, cv2.CV_8U, 0, 1)
                g_y = np.mean(grad_y) / 255 * 100
                print g_y

                #calculate of the gradient in x axis
                grad_x = mat_temp.copy()
                grad_x = cv2.Sobel(grad_x, cv2.CV_8U, 1, 0)
                g_x = np.mean(grad_x) / 255 * 100
                print g_x

                #calculate the end point of the vector, with the gradients
                dest = (min_x + x / 2 + int(g_x), min_y + x / 2 + int(g_y))

                #draw the line
                cv2.line(img_grad, center, dest, (255,0,0),  thickness=2, lineType=8, shift=0)

    img_gradVect = img_grad.copy()

    if(grad_vectors_mode):
        cv2.imshow('CS512_HW2', img_grad)

img_save = []
img_mysmooth = []
img_gradVect = []

cv2_smooth_mode = False  # variable to enable the use of the trackbar s
my_smooth_mode = False  # variable to enable the use of the trackbar S
cv2_rotation_mode = False  # variable to enable the use of the trackbar r
grad_vectors_mode = False  # variable to enable the use of the trackbar p

def main():

    global img, img_save, img_mysmooth, img_gradVect
    global cv2_smooth_mode, my_smooth_mode, cv2_rotation_mode, grad_vectors_mode

    cv2.namedWindow('CS512_HW2')

    if len(sys.argv) <= 1:
        img = camera_capture()
    else :
        img = cv2.imread(sys.argv[1])

    # img = cv2.imread('hw2_bioshock_img.jpg')

    if img is None:
        print "Image not found"
        return 0


    img_save = img.copy()
    img_mysmooth = img.copy()
    img_gradVect = img.copy()

    cv2.createTrackbar('s', 'CS512_HW2', 0, 30, cv2_smooth)
    cv2.createTrackbar('S', 'CS512_HW2', 0, 3, my_smooth)
    cv2.createTrackbar('r', 'CS512_HW2', 0, 360, cv2_rotate)
    cv2.createTrackbar('p', 'CS512_HW2', 0, 360, gradient_field)

    key = 0
    index_color = 0

    b, g, r = cv2.split(img)
    col_nul = r.copy()
    col_nul[:] = 0
    first_call_color = True  # used to get the rgb components the first time 'c' is entered

    skipped_img_display = False

    cv2.imshow('CS512_HW2', img)

    while (key != 27):

        if skipped_img_display :
            skipped_img_display = False
        else :
            cv2.imshow('CS512_HW2', img)

        key = cv2.waitKey(0)
        print("key : %d" % key)

        if not first_call_color and key != ord('c'):
            first_call_color = True

        if (cv2_smooth_mode):
            cv2_smooth_mode = False
            x_s = cv2.getTrackbarPos("s", "CS512_HW2")
            x_s = 2 * x_s + 1
            img = cv2.blur(img, (x_s, x_s))
        elif (my_smooth_mode):
            my_smooth_mode = False
            img = img_mysmooth.copy() #use of a global variable because the computation time is very long
        elif (cv2_rotation_mode):
            cv2_rotation_mode = False
            if (len(img.shape) != 3):
                print("not enough channels to convert to grey")
                img_rot = img.copy()
            else:
                img_rot = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cols = img_rot.shape[0]
            rows = img_rot.shape[1]
            center = (rows / 2, cols / 2)
            x_r = cv2.getTrackbarPos("r", "CS512_HW2")
            mat_rot= cv2.getRotationMatrix2D(center, x_r, 1.0)
            img_rotated = cv2.warpAffine(img_rot, mat_rot, (rows, cols))
            img = img_rotated.copy()
        elif (grad_vectors_mode) :
            grad_vectors_mode = False
            img = img_gradVect.copy()

        if key == 27:  # wait for ESC key to exit
            print("esc taped : exit")

        elif key == ord('i'):  # wait for 'i' key to reload the file and exit
            print("i taped : reload")
            img = img_save

        elif key == ord('w'):  # wait for 'w' key to save the current img
            print("w taped : save")
            cv2.imwrite('out.png', img)

        elif key == ord('g'):  # wait for 'g' key to convert to greyscale with opencv function
            print("g taped : cv grey")
            if (len(img.shape) != 3):
                print("not enough channels to convert to grey")
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        elif key == ord('c'):  # wait for 'c' key to cycle through the color channels
            print("c taped : color chanels")
            if (len(img.shape) != 3):
                print("not enough channels to change colors")
            else:
                # at the first iteration, init the variables
                if(first_call_color):
                    b, g, r = cv2.split(img)
                    col_nul = r.copy()
                    col_nul[:] = 0
                    first_call_color = False

                # select the channel to filter
                img = cv2.merge((b, g, r))
                if (index_color == 0):
                    img = cv2.merge((b, col_nul, col_nul))
                elif (index_color == 1):
                    img = cv2.merge((col_nul, g, col_nul))
                else:
                    img = cv2.merge((col_nul, col_nul, r))

                #update index
                index_color = index_color + 1 if index_color < 2 else 0

        elif key == ord('s'):  # wait for 's' key to convert to greyscale and smooth it
            print("s taped : cv2 smooth")
            if (len(img.shape) != 3):
                print("not enough channels to convert to grey")
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            x_s = cv2.getTrackbarPos("s", "CS512_HW2")
            x_s = 2 * x_s + 1
            img = cv2.blur(img, (x_s, x_s))
            cv2.imshow('CS512_HW2', img)
            skipped_img_display = True
            cv2_smooth_mode = True

        elif key == ord('d'):  # wait for 'd' key to downsample the image by a factor of 2
            print("d taped : downsample by two")
            if img.shape[0] < 10 or img.shape[1] < 10:
                print("smaller size reach")
            else:
                rows = img.shape[1] // 2
                cols = img.shape[0] // 2
                img = cv2.resize(img, (rows, cols), interpolation=cv2.INTER_NEAREST)

        elif key == ord('x'):  # wait for 'x' key to make a convolution with an x derivative filter
            print("x taped : x derivative filter")
            if len(img.shape) != 3:
                print("not enough channels to convert to grey")
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.Sobel(img, cv2.CV_8U, 1, 0)
            scale = np.max(img) / 255
            img = (img / scale).astype(np.uint8)

        elif key == ord('y'):  # wait for 'y' key to make a convolution with an y derivative filter
            print("y taped : y derivative filter")
            if len(img.shape) != 3:
                print("not enough channels to convert to grey")
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.Sobel(img, cv2.CV_8U, 0, 1)
            scale = np.max(img) / 255
            img = (img / scale).astype(np.uint8)

        elif key == ord('m'): # wait for 'm' key to show the magnitude of the image
            print("m taped : magnitude")
            if len(img.shape) != 3:
                print("not enough channels to convert to grey")
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            dx =  cv2.Sobel(img, cv2.CV_64F, 1, 0)
            dy =  cv2.Sobel(img, cv2.CV_64F, 0, 1)
            img = np.sqrt(dx**2 + dy**2)
            scale = np.max(img) / 255
            img = (img / scale).astype(np.uint8)

        elif key == ord('p'): # wait for 'p' key to show the gradient vectors
            print("p taped : grad vectors")
            grad_vectors_mode = True
            skipped_img_display = True

        elif key == ord('r'):
            print("r taped : rotation")
            cv2_rotation_mode = True
            if (len(img.shape) != 3):
                print("not enough channels to convert to grey")
                img_rot = img.copy()
            else:
                img_rot = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cols = img_rot.shape[0]
            rows = img_rot.shape[1]
            center = (rows / 2, cols / 2)
            x_r = cv2.getTrackbarPos("r", "CS512_HW2")
            mat_rot= cv2.getRotationMatrix2D(center, x_r, 1.0)
            img_rotated = cv2.warpAffine(img_rot, mat_rot, (rows, cols))
            cv2.imshow("CS512_HW2", img_rotated)
            skipped_img_display = True

        elif key == ord('h'):
            print("h taped : help")
            help_img = cv2.imread('../data/help_menu.png')
            cv2.imshow("CS512_HW2_help",help_img)

        elif key == 225 or key == 226:  # Uppercase key

            key_uppercase = cv2.waitKey(0)

            if key_uppercase == ord('g'):  # wait for 'G' key to convert to greyscale with my function
                print("G taped : my grey")
                if (len(img.shape) != 3):
                    print("not enough channels to convert to grey")
                else:
                    coefficients = [0.114, 0.587, 0.299] #matrice with common coef for conversion to grey
                    m_grey = np.array(coefficients).reshape((1, 3))
                    img = cv2.transform(img, m_grey)

            elif key_uppercase == ord('s'):  # wait for 'S' key to convert to greyscale and smooth it
                print("S taped : my smooth")
                my_smooth_mode = True

                if (len(img.shape) != 3):
                    print("not enough channels to convert to grey")
                    img_smooth = img.copy()
                else:
                    img_smooth = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                x_S = cv2.getTrackbarPos("S", "CS512_HW2")
                if x_S == 0:
                    print("too small to apply a visible smoothing")
                else :
                    # the 3 user input correspond to these sizes:
                    #  1 = 3x3 matrice
                    #  2 = 5x5 matrice
                    #  3 = 7x7 matrice
                    x_S = 3 if x_S == 1 else (5 if x_S == 2 else 7)

                    img_smoothed = img_smooth.copy()

                    for ind_x in range(x_S + 2, img_smooth.shape[1] - x_S - 2):
                        for ind_y in range(x_S + 2, img_smooth.shape[0] - x_S - 2):
                            mat_temp = img_smooth[ind_y - (x_S - 1) / 2:ind_y + (x_S - 1) / 2 + 1,
                                       ind_x - (x_S - 1) / 2:ind_x + (x_S - 1) / 2 + 1]
                            img_smoothed[ind_y][ind_x] = np.sum(mat_temp / (x_S ** 2))

                    # Only for this one, we pu the result in a global varaible, so we don't have to do the computation again
                    img_mysmooth = img_smoothed.copy()

                    cv2.imshow('CS512_HW2', img_smoothed)

                    skipped_img_display = True

            elif key_uppercase == ord('d'):  # wait for 'D' key to downsample the imag by a factor of 2 and blur it
                print("D taped : downsample by two and blur")
                if img.shape[0] < 10 or img.shape[1] < 10 :
                    print("smaller size reach")
                else:
                    rows = img.shape[1] // 2
                    cols = img.shape[0] // 2
                    img = cv2.resize(img, (rows, cols), interpolation=cv2.INTER_NEAREST)
                    img = cv2.blur(img, (3, 3))


    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
