import sys
import traceback
import tellopy
import av
import cv2.cv2 as cv2  # for avoidance of pylint error
import numpy
import time
import os
import datetime
import imutils
import numpy as np
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", default='./resnet10/deploy.prototxt.txt',
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model",    default='./resnet10/res10_300x300_ssd_iter_140000.caffemodel',
    help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
ap.add_argument("-s", "--save",  action='store_true',
    help="save the video")
args = vars(ap.parse_args())

def handleFileReceived(event, sender, data):
    global date_fmt
    # Create a file in ~/Pictures/ to receive image data from the drone.
    path = '%s/tello-%s.jpeg' % (
        os.getenv('HOMEPATH'),                              #Changed from Home to Homepath
        datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S'))
    with open(path, 'wb') as fd:
        fd.write(data)
    #print('Saved photo to ',path)

if args["save"]:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc, 20.0, (400,300))

net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

def main():
    drone = tellopy.Tello()
    landed = True
    speed = 30
    up,down,left,right,forw,back,clock,ctclock = False,False,False,False,False,False,False,False
    ai = True
    pic360 = False
    currentPic = 0
    move360 = False
    try:
        drone.connect()
        drone.wait_for_connection(60.0)

        container = av.open(drone.get_video_stream())
        drone.subscribe(drone.EVENT_FILE_RECEIVED, handleFileReceived)
        # skip first 300 frames
        frame_skip = 300
        while True:
            try:
                for frame in container.decode(video=0):
                    if 0 < frame_skip:
                        frame_skip = frame_skip - 1
                        continue
                    start_time = time.time()
                    image = cv2.cvtColor(numpy.array(frame.to_image()), cv2.COLOR_RGB2BGR)

                    image = imutils.resize(image, width=400)
                    (h, w) = image.shape[:2]
                    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                    (300, 300), (104.0, 177.0, 123.0))
                    net.setInput(blob)
                    detections = net.forward()

                    face_dict = {}

                    for i in range(0, detections.shape[2]):
                        # extract the confidence (i.e., probability) associated with the
                        # prediction
                        confidence = detections[0, 0, i, 2]

                        # filter out weak detections by ensuring the `confidence` is
                        # greater than the minimum confidence
                        if confidence < 0.5:
                            continue

                        # compute the (x, y)-coordinates of the bounding box for the
                        # object
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        # draw the bounding box of the face along with the associated
                        # probability
                        text = "{:.2f}%".format(confidence * 100)
                        face_dict[text]=box

                    # Will go to face with the highest confidence
                    try:    
                        H,W,_ = image.shape
                        distTolerance = 0.05 * np.linalg.norm(np.array((0, 0))- np.array((w, h)))

                        box = face_dict[sorted(face_dict.keys())[0]]
                        y = startY - 10 if startY - 10 > 10 else startY + 10
                        cv2.rectangle(image, (startX, startY), (endX, endY),
                            (0, 0, 255), 2)

                        distance = np.linalg.norm(np.array((startX,startY))-np.array((endX,endY)))

                        if int((startX+endX)/2) < W/2-distTolerance :
                            #print('CounterClock')
                            drone.counter_clockwise(30)
                            ctclock = True
                        elif int((startX+endX)/2) > W/2+distTolerance:
                            #print('Clock')
                            drone.clockwise(30)
                            clock = True
                        else:
                            if ctclock:
                                drone.counter_clockwise(0)
                                ctclock = False
                                #print('CTClock 0')
                            if clock:
                                drone.clockwise(0)
                                clock = False
                                #print('Clock 0')
                        
                        if int((startY+endY)/2) < H/2-distTolerance :
                            drone.up(30)
                            #print('Up')
                            up = True
                        elif int((startY+endY)/2) > H/2+distTolerance :
                            drone.down(30)
                            #print('Down')
                            down = True
                        else:
                            if up:
                                up = False
                                #print('Up 0')
                                drone.up(0)

                            if down:
                                down = False
                                #print('Down 0')
                                drone.down(0)

                        #print(int(distance))

                        if int(distance) < 110-distTolerance  :
                            forw = True
                            #print('Forward')
                            drone.forward(30)
                        elif int(distance) > 110+distTolerance :
                            drone.backward(30)
                            #print('Backward')
                            back = True
                        else :
                            if back:
                                back = False
                                #print('Backward 0')
                                drone.backward(0)
                            if forw:
                                forw = False
                                #print('Forward 0')
                                drone.forward(0)
                            

                    except Exception as e:
                        #print(e)
                        None

                    if args["save"]:
                            out.write(image)

                    cv2.imshow('Original', image)

                    #cv2.imshow('Canny', cv2.Canny(image, 100, 200))
                    if frame.time_base < 1.0/60:
                        time_base = 1.0/60
                    else:
                        time_base = frame.time_base
                    frame_skip = int((time.time() - start_time)/time_base)
                    keycode = cv2.waitKey(1)
                    
                    if keycode == 32 :
                        if landed:
                            drone.takeoff()
                            landed = False
                        else:
                            drone.land()
                            landed = True

                    if keycode == 27 :
                        raise Exception('Quit')

                    if keycode == 13 :
                        drone.take_picture()
                        time.sleep(0.25)
                        #pic360 = True
                        #move360 = True

                    if keycode & 0xFF == ord('q') :
                        pic360 = False
                        move360 = False 

            except Exception as e:
                print(e)
                break
                     

    except Exception as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print(ex)
    finally:
        if args["save"]:
            out.release()
        drone.quit()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
