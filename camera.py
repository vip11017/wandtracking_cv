# %%
from picamera2 import Picamera2
import cv2
import numpy as np

# %%
picam = Picamera2()
picam.configure()
picam.start()

# %%
canvas = None
prev_point = None

no_blob_count = 0
max_no_blob_count = 50

params = cv2.SimpleBlobDetector_Params()
# Filter by Area.
params.filterByArea = True
params.maxArea = 500

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.68

#Filter by Color
params.filterByColor = False
params.blobColor = 255

detector = cv2.SimpleBlobDetector_create(params)    


while True:
    img = picam.capture_array()
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    #cv2.imshow("Live Camera Feed (IR)", img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    res, thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)
    #cv2.imshow("Threshold", thresh)

    keypoints = detector.detect(thresh)
    test = cv2.drawKeypoints(img, keypoints, None, (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #cv2.imshow("SimpleBlobDetector", test)

    if canvas is None:
        canvas = np.zeros_like(img)
        canvas = cv2.bitwise_not(canvas)

    if keypoints:
        no_blob_count = 0
        curr_point = (int(keypoints[0].pt[0]), int(keypoints[0].pt[1]))

        # Draw a line or point
        if prev_point:
            cv2.line(canvas, prev_point, curr_point, (0, 0, 0), 2)  # Draw line
        else:
            cv2.circle(canvas, curr_point, 2, (0, 0, 0), -1)  # Draw initial point

        prev_point = curr_point  # Update the previous point

    else:
        no_blob_count += 1
        if no_blob_count > max_no_blob_count:
            no_blob_count = 0
            prev_point = None
            canvas = np.zeros_like(img)
            canvas = cv2.bitwise_not(canvas)

    output = cv2.addWeighted(img, 0.5, canvas, 0.5, 0)
    cv2.imshow("Final", output)
    
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imshow("Canvas Image", canvas)
        cv2.imwrite('live_images/pre_air_drawing.png', canvas)

        canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGRA2GRAY)
        non_white_pixels = np.where(canvas_gray < 255)

        if non_white_pixels[0].size > 0 and non_white_pixels[1].size > 0:
            # Calculate bounding box
            x_min = np.min(non_white_pixels[1])
            x_max = np.max(non_white_pixels[1])
            y_min = np.min(non_white_pixels[0])
            y_max = np.max(non_white_pixels[0])

            # Define a border size (padding)
            border = 10  # Adjust this value to control the border size

            # Add padding and ensure the coordinates stay within the canvas boundaries
            x_min = max(x_min - border, 0)
            x_max = min(x_max + border, canvas.shape[1])
            y_min = max(y_min - border, 0)
            y_max = min(y_max + border, canvas.shape[0])

            cropped_canvas = canvas[y_min:y_max, x_min:x_max]


            # Save the cropped image
            cv2.imwrite('live_images/cropped_drawing.png', cropped_canvas)
            print('Cropped image saved as cropped_drawing.png')
        else:
            print("No drawing detected to crop.")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
picam.stop()
cv2.destroyAllWindows()
