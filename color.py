import cv2
import numpy as np
import datetime

import functions as f

'''
CODES
    esc - close
    f - fullscreen
    p - pause
    r - reset
    s - save

    g - green
    u - purple
    b - blue
    a - all

    m - mirror
    l - line
'''

debug_mode = False
fullscreen_mode = False
paused = False
changed_mode = True

filter_mode = 0
line_mode = 'canny'
inverted_color_mode = False

color_mode = 'green'
color_limits = (70, 85)

min_saturation = 90
min_brightness = 90

l_h, l_s, l_v = 15, min_saturation, min_brightness
u_h, u_s, u_v = 160, 255, 255

def nothing(x):
    pass

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


print("Frame default resolution: (" + str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) + "; " + str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) + ")")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # 1024x576; 800x600;
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360) # 16:9 -> 640x480; 960x540; 854x480; 640x360
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

drawing = np.zeros((frame.shape[0], frame.shape[1], 3), dtype='uint8')

cv2.namedWindow("result", cv2.WINDOW_NORMAL)
cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
cv2.namedWindow("mask", cv2.WINDOW_NORMAL)

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # -=-=-=-=-=-
    #   Filters
    # -=-=-=-=-=-

    match filter_mode:

        case 0:
            frame = cv2.GaussianBlur(frame, (5,5), 0)

        case 1:
            frame = f.apply_mirror(frame)
            frame = cv2.GaussianBlur(frame, (5,5), 0)

        case 2:
            frame = cv2.GaussianBlur(frame, (5,5), 0)
            frame = f.apply_pixelation(frame)

        case 3:
            frame = f.apply_fisheye(frame)
            frame = cv2.GaussianBlur(frame, (5,5), 0)

        case 4:
            frame = cv2.flip(frame, 0)
            frame = cv2.GaussianBlur(frame, (5,5), 0)

    # -=-=-=-=-=-
    # Processing
    # -=-=-=-=-=-

    # MASK

    #cv2.imshow('blur', blurred_frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_blue = np.array([l_h, l_s, l_v])
    upper_blue = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, lower_blue, upper_blue) 

    # COLORS
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_hsv[:, :, 1] = 255  # Max saturation
    frame_hsv[:, :, 2] = 255  # Max brightness
    result = cv2.cvtColor(frame_hsv, cv2.COLOR_HSV2BGR)

    result = cv2.bitwise_and(result, result, mask=mask)

    # COMBINE ALL
    drawing = cv2.add(result, drawing)

    if line_mode == 'canny':
        result = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = cv2.Canny(result, 150, 175)
        result = cv2.merge([result, result, result])

    else:
        result = np.zeros((frame.shape[0], frame.shape[1], 3), dtype='uint8')
    
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
            debug_mode = not debug_mode
            changed_mode = True
        
        case 102: # f code
            fullscreen_mode = not fullscreen_mode
            changed_mode = True

        case 112: # p code
            paused = 1 - paused
        
        case 114: # r code
            drawing = np.zeros((frame.shape[0], frame.shape[1], 3), dtype='uint8')

        case 115: # s code

            logo = cv2.imread('manna_team_logo.png', cv2.IMREAD_UNCHANGED)
            logo = f.rescaleFrame(logo, 0.5)
            frame_with_logo = f.rescaleFrame(result, 2).copy()
            logo_height, logo_width, _ = logo.shape
            roi = frame_with_logo[10:10+logo_height, 10:10+logo_width]
            logo_mask = logo[:, :, 3]
            logo = logo[:, :, 0:3]
            logo = cv2.bitwise_and(logo, logo, mask=logo_mask)
            logo = cv2.add(roi, logo)
            frame_with_logo[10:10+logo_height, 10:10+logo_width] = logo

            if inverted_color_mode: # gambiarra
                frame_with_logo = cv2.bitwise_not(frame_with_logo)

            current_datetime = datetime.datetime.now()
            timestamp = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

            if cv2.imwrite(f'print_{timestamp}.png', frame_with_logo):
                print('Imagem salva com sucesso!')
            else:
                print('Erro ao salvar a imagem.')


            cv2.imshow('windows', frame_with_logo)

        # COLOR CODES

        case 49: # 1 code [verde]
            color_mode = 'verde'
            color_limits = (70, 85)
            changed_mode = True
        
        case 50: # 2 code [azul claro]
            color_mode = 'azul claro'
            color_limits = (95, 110)
            changed_mode = True

        case 51: # 3 code [azul escuro]
            color_mode = 'azul escuro'
            color_limits = (100, 115)
            changed_mode = True
        
        case 52: # 4 code [roxo]
            color_mode = 'roxo'
            color_limits = (150, 165)
            changed_mode = True

        case 48: # 0 code [entre verde e roxo]
            color_mode = 'all'
            color_limits = (70, 165)
            changed_mode = True

        # FILTER CODES

        case 105: # i code
            inverted_color_mode = not inverted_color_mode

        case 98: # b code
            filter_mode = (filter_mode+1) % 5

        case 108: # l code
            line_mode = 'canny' if line_mode == None else None

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

        l_s = min_saturation
        l_v = min_brightness

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

    if debug_mode:

        # FPS
        fps = cv2.getTickFrequency()/(cv2.getTickCount()-timer)
        fps = f'fps: {int(fps)}'
        cv2.putText(mask,fps,(mask.shape[1]-100,50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
        cv2.putText(mask,'debug mode',(50,mask.shape[0]-50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

        timer = cv2.getTickCount() # must be before video read

        # SHOW
        cv2.imshow("frame", frame)
        # cv2.imshow("drawing", drawing)

    else:
        cv2.putText(mask,f'color mode: {color_mode}',(50,mask.shape[0]-50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

    cv2.imshow("mask", mask)

    # COLOR CONTOUR
    if color_mode != 'all':
        color_limits_average = (color_limits[0]+color_limits[1]) / 2
        result = f.add_hsv_border(result, (color_limits_average, 255, 255))

    # INVERTED COLOR MODE
    if inverted_color_mode:
        result = cv2.bitwise_not(result)

    # FULLSCREEN MODE
    if fullscreen_mode:
        cv2.setWindowProperty("result", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.setWindowProperty("result", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    cv2.imshow("result", result)

    if changed_mode:
        try:
            cv2.destroyWindow('frame')
            cv2.destroyWindow('drawing')

            cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
            cv2.namedWindow("drawing", cv2.WINDOW_NORMAL)

        except:
            pass

    changed_mode = False

cap.release()
cv2.destroyAllWindows()