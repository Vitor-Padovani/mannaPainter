import cv2
import numpy as np
import datetime

debug_mode = False
paused = False
changed_mode = True

filter_mode = None
line_mode = 'canny'

color_mode = 'green'
color_limits = (70, 85)

min_saturation = 90
min_brightness = 90

l_h, l_s, l_v = 15, min_saturation, min_brightness
u_h, u_s, u_v = 160, 255, 255

def nothing(x):
    pass

def rescaleFrame(frame, scale=0.5):
    width = int(frame.shape[1] * scale) # don't panic! just calculating
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)

    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


print("Frame default resolution: (" + str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) + "; " + str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) + ")")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800) # 1024x576; 800x600; 640x480
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
# drawing_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype='uint8')
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

    # MASK

    blurred_frame = cv2.GaussianBlur(frame, (5,5), 0)
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
    
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
        result = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2GRAY)
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
            debug_mode = 1 - debug_mode
            changed_mode = True

        case 112: # p code
            paused = 1 - paused
        
        case 114: # r code
            drawing = np.zeros((frame.shape[0], frame.shape[1], 3), dtype='uint8')

        case 115:
            logo = cv2.imread('C:/code/openCV/painter/hsvMask/manna_team_logo.png', cv2.IMREAD_UNCHANGED)
            logo = rescaleFrame(logo, 0.5)
            frame_with_logo = result.copy()
            logo_height, logo_width, _ = logo.shape
            roi = frame_with_logo[10:10+logo_height, 10:10+logo_width]
            logo_mask = logo[:, :, 3]
            logo = logo[:, :, 0:3]
            logo = cv2.bitwise_and(logo, logo, mask=logo_mask)
            logo = cv2.add(roi, logo)
            frame_with_logo[10:10+logo_height, 10:10+logo_width] = logo

            current_datetime = datetime.datetime.now()
            timestamp = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

            if cv2.imwrite(f'print_{timestamp}.png', frame_with_logo):
                print('Imagem salva com sucesso!')
            else:
                print('Erro ao salvar a imagem.')


            cv2.imshow('windows', frame_with_logo)

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
        cv2.imshow("drawing", drawing)

    else:
        cv2.putText(mask,f'color mode: {color_mode}',(50,mask.shape[0]-50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)


    cv2.imshow("mask", mask)
    cv2.imshow("result", result)

    if changed_mode:
        try:
            cv2.destroyWindow('frame')
            cv2.destroyWindow('drawing')

        except:
            pass

    changed_mode = False

cap.release()
cv2.destroyAllWindows()