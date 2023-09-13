import cv2
import numpy as np

debug_mode = False
paused = False
changed_mode = False

filter_mode = None

color_mode = 'all'
color_limits = (15, 160)

l_h, l_s, l_v = 15, 100, 100
u_h, u_s, u_v = 160, 255, 255

def nothing(x):
    pass

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

print("Frame default resolution: (" + str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) + "; " + str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) + ")")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
print("Frame resolution set to: (" + str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) + "; " + str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) + ")")

cv2.namedWindow("Trackbars")
cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

timer = cv2.getTickCount() # must be before video read
_, frame = cap.read()

# TODO make the drawing receive the drawing_mask but with the color blue
drawing_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype='uint8')
drawing = np.zeros((frame.shape[0], frame.shape[1], 3), dtype='uint8')

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # -=-=-=-=-=-
    #   Filters
    # -=-=-=-=-=-

    match filter_mode:

        case 'flip':
            frame = cv2.flip(frame, 1)
            half = frame[:frame.shape[0], :frame.shape[1]//2]
            frame[:, frame.shape[1]//2:] = cv2.flip(half, 1)

    # -=-=-=-=-=-
    # Processing
    # -=-=-=-=-=-

    blurred_frame = cv2.GaussianBlur(frame, (5,5), 0)
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
    
    lower_blue = np.array([l_h, l_s, l_v])
    upper_blue = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)


    drawing_mask = cv2.bitwise_or(drawing_mask, mask)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    drawing = cv2.add(result, drawing)

    result = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2GRAY)
    result = cv2.Canny(result, 150, 175)
    result = cv2.merge([result, result, result])
    result = cv2.add(drawing, result)

    # -=-=-=-=-
    #   KEYS
    # -=-=-=-=-

    if paused:
        key = cv2.waitKey(0) # waits for something
        print(key)

    else:
        key = cv2.waitKey(1) # returns -1 if nothing

    match key:

        # SYSTEM CODES

        case 27: # esc code
            break

        case 100: # d code
            debug_mode = 1 - debug_mode
            changed_mode = True

        case 112: # p code
            paused = 1 - paused
        
        case 114: # r code
            drawing = np.zeros((frame.shape[0], frame.shape[1], 3), dtype='uint8')

        # COLOR CODES

        case 103: # g code [verde]
            color_mode = 'green'
            color_limits = (70, 85)
            changed_mode = True

        case 121: # y code [amarelo]
            color_mode = 'yellow'
            color_limits = (15, 30)
            changed_mode = True
        
        case 98: # b code [azul claro]
            color_mode = 'light blue'
            color_limits = (95, 110)
            changed_mode = True

        case 97: # a code [todas menos vermelho]
            color_mode = 'all'
            color_limits = (15, 115)
            changed_mode = True

        # FILTER CODES

        case 102: # f code
            filter_mode = 'flip' if filter_mode == None else None

    if debug_mode:
        l_h = cv2.getTrackbarPos("L - H", "Trackbars")
        l_s = cv2.getTrackbarPos("L - S", "Trackbars")
        l_v = cv2.getTrackbarPos("L - V", "Trackbars")
        u_h = cv2.getTrackbarPos("U - H", "Trackbars")
        u_s = cv2.getTrackbarPos("U - S", "Trackbars")
        u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    else:
        l_h = color_limits[0]
        u_h = color_limits[1]

        l_s = 100
        l_v = 100

        if changed_mode:
            cv2.setTrackbarPos("L - H", "Trackbars", l_h)
            cv2.setTrackbarPos("L - S", "Trackbars", l_s)
            cv2.setTrackbarPos("L - V", "Trackbars", l_v)
            cv2.setTrackbarPos("U - H", "Trackbars", u_h)
            cv2.setTrackbarPos("U - S", "Trackbars", u_s)
            cv2.setTrackbarPos("U - V", "Trackbars", u_v)

    # -=-=-=-=-
    #    UI
    # -=-=-=-=-

    # FPS
    if debug_mode:
        fps = cv2.getTickFrequency()/(cv2.getTickCount()-timer)
        fps = f'fps: {int(fps)}'
        cv2.putText(mask,fps,(mask.shape[1]-100,50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
        cv2.putText(mask,'debug mode',(50,mask.shape[0]-50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

        timer = cv2.getTickCount() # must be before video read

        cv2.imshow("frame", frame)
    
    else:
        cv2.putText(mask,f'color mode: {color_mode}',(50,mask.shape[0]-50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)


    cv2.imshow("mask", mask)
    cv2.imshow("result", result)

    if changed_mode:
        try:
            cv2.destroyWindow('frame')
        except:
            pass
        changed_mode = False

cap.release()
cv2.destroyAllWindows()