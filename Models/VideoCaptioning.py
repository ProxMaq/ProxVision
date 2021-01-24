
import time
import cv2

def startCamera(skip_seconds):
    cameraStream = cv2.VideoCapture(0)

    # Get frame information
    success, frame = cameraStream.read()
    height, width, layers = frame.shape
    new_h = round(height / 2)
    new_w = round(width / 2)

    # Get frames per seconds.
    fps = int(cameraStream.get(cv2.CAP_PROP_FPS))
    #CV_CAP_PROP_POS_FRAMES

    frame_count = 0

    skip_seconds
    # Empty string to fetch caption sentence output from model
    caption_text = ''
    while (cameraStream.isOpened()):
        success, frame = cameraStream.read()
        if success == True:
            resized_frame = cv2.resize(frame, (new_w, new_h))

            # @Harsh uncomment below line and add your 'predict_caption()' method here.
            # caption_text = model.predict_caption(resized_frame)
            print(f'{caption_text}') # later 'caption_text' will be converted into audio.

            # Commented, because hour task is to generate caption (and later convert it into audio)
            # cv2.imshow('Video', frame)

            if cv2.waitKey(fps) & 0xFF == ord('q'):
                print('Key \'Q\' is pressed !')
                break
        else:
            print('Video Capture returns FALSE !')
            break

        frame_count += fps * skip_seconds
        cameraStream.set(1, frame_count)
        time.sleep(skip_seconds)

    # Release camera resource
    cameraStream.release()
    cv2.destroyAllWindows()
    print('Application exited !')

def runApp():
    print('Application started !')
    startCamera(3)

def main():
    runApp()
    
if __name__ == '__main__':
    main()