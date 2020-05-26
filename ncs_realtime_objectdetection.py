# USAGE
# python ncs_realtime_objectdetection.py --graph graphs/mobilenetgraph --display 1
# python ncs_realtime_objectdetection.py --graph graphs/mobilenetgraph --confidence 0.9 --display 1

# import the necessary packages
from mvnc import mvncapi as mvnc
from imutils.video import VideoStream
from imutils.video import FPS
import paho.mqtt.publish as publish
import time
import argparse
import numpy as np
import cv2

# initialize the list of class labels our network was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ("background", "aeroplane", "bicycle", "bird",
    "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor")
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# frame dimensions should be sqaure
PREPROCESS_DIMS = (300, 300)

# Used Variables
person = 0
tv = 0
CanSwitch = False
OPEN = False

def preprocess_image(input_image):
    # preprocess the image
    preprocessed = cv2.resize(input_image, PREPROCESS_DIMS)
    preprocessed = preprocessed - 127.5
    preprocessed = preprocessed * 0.007843
    preprocessed = preprocessed.astype(np.float16)

    # return the image to the calling function
    return preprocessed

def predict(image, graph):
    # preprocess the image
    image = preprocess_image(image)

    # send the image to the NCS and run a forward pass to grab the
    # network predictions
    graph.LoadTensor(image, 'My Prediction')

    output, userobj = graph.GetResult()

    # grab the number of valid object predictions from the output,
    # then initialize the list of predictions
    num_valid_boxes = output[0]
    predictions = []

    # loop over results
    for box_index in range(int(num_valid_boxes)):
        # calculate the base index into our array so we can extract
        # bounding box information
        base_index = 7 + box_index * 7

        # boxes with non-finite (inf, nan, etc) numbers must be ignored
        if (not np.isfinite(output[base_index]) or
            not np.isfinite(output[base_index + 1]) or
            not np.isfinite(output[base_index + 2]) or
            not np.isfinite(output[base_index + 3]) or
            not np.isfinite(output[base_index + 4]) or
            not np.isfinite(output[base_index + 5]) or
            not np.isfinite(output[base_index + 6])):
            continue

        # extract the image width and height and clip the boxes to the
        # image size in case network returns boxes outside of the image
        # boundaries
        (h, w) = image.shape[:2]
        x1 = max(0, int(output[base_index + 3] * w))
        y1 = max(0, int(output[base_index + 4] * h))
        x2 = min(w, int(output[base_index + 5] * w))
        y2 = min(h, int(output[base_index + 6] * h))

        # grab the prediction class label, confidence (i.e., probability),
        # and bounding box (x, y)-coordinates
        pred_class = int(output[base_index + 1])
        pred_conf = output[base_index + 2]
        pred_boxpts = ((x1, y1), (x2, y2))

        # create prediciton tuple and append the prediction to the
        # predictions list
        prediction = (pred_class, pred_conf, pred_boxpts)
        predictions.append(prediction)

    # return the list of predictions to the calling function
    return predictions

def open_ncs_device():
    # grab a list of all NCS devices plugged in to USB
    print("[INFO] finding NCS devices...")
    devices = mvnc.EnumerateDevices()

    # if no devices found, exit the script
    if len(devices) == 0:
        print("[INFO] No devices found. Please plug in a NCS")
        quit()

    # use the first device since this is a simple test script
    # (you'll want to modify this is using multiple NCS devices)
    print("[INFO] found {} devices. device0 will be used. "
        "opening device0...".format(len(devices)))
    device = mvnc.Device(devices[0])
    device.OpenDevice()

    return device

def load_graph(device):
    # open the graph file
    print("[INFO] loading the graph file into RPi memory...")
    with open(args["graph"], mode="rb") as f:
        graph_file = f.read()

    # load the graph into the NCS
    print("[INFO] allocating the graph on the NCS...")
    graph = device.AllocateGraph(graph_file)

    return graph

def close_ncs_device():
    # stop the FPS counter timer
    fps.stop()

    # destroy all windows if we are displaying them
    if args["display"] > 0:
        cv2.destroyAllWindows()

    # stop the video stream
    vs.stop()

    # clean up the graph and device
    graph.DeallocateGraph()
    device.CloseDevice()

    # display FPS information
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-g", "--graph", required=True,
    help="path to input graph file")
ap.add_argument("-c", "--confidence", default=.9,
    help="confidence threshold")
ap.add_argument("-d", "--display", type=int, default=0,
    help="switch to display image on screen")
args = vars(ap.parse_args())

# open a pointer to the video stream thread and allow the buffer to
# start to fill, then start the FPS counter

device = open_ncs_device()
graph = load_graph(device)

print("[INFO] starting the video stream and FPS counter...")
vs = VideoStream(usePiCamera=True).start()
time.sleep(1)
fps = FPS().start()

# loop over frames from the video file stream
while True:
    try:
        # grab the frame from the threaded video stream
        # make a copy of the frame and resize it for display/video purposes
        frame = vs.read()
        image_for_result = frame.copy()
        image_for_result = cv2.resize(image_for_result, PREPROCESS_DIMS)

        # use the NCS to acquire predictions
        predictions = predict(frame, graph)

        inference_time = graph.GetGraphOption( mvnc.GraphOption.TIME_TAKEN )

        # loop over our predictions
        for (i, pred) in enumerate(predictions):
            # extract prediction data for readability
            (pred_class, pred_conf, pred_boxpts) = pred

            # filter out weak detections by ensuring the `confidence`
            # is greater than the minimum confidence
            if pred_conf > args["confidence"]:
                # print prediction to terminal
                print("[INFO] Prediction #{}: class={}, confidence={}, "
                    "boxpoints={}".format(i, CLASSES[pred_class], pred_conf,
                    pred_boxpts))

                print( "I found these objects in "
                + " ( %.2f ms ):" % ( np.sum( inference_time ) ) )

                # check if we should show the prediction data
                # on the frame
                if args["display"] > 0:
                    # build a label consisting of the predicted class and
                    # associated probability
                    label = "{}: {:.2f}%".format(CLASSES[pred_class],
                        pred_conf * 100)

                    # extract information from the prediction boxpoints
                    (ptA, ptB) = (pred_boxpts[0], pred_boxpts[1])
                    ptA = (ptA[0] , ptA[1]) #1st coordenate
                    ptB = (ptB[0] , ptB[1]) #2nd coordenate
                    (startX, startY) = (ptA[0], ptA[1])
                    y = startY - 15 if startY - 15 > 15 else startY + 15

                    # display the rectangle and label text
                    cv2.rectangle(image_for_result, ptA, ptB,
                        COLORS[pred_class], 2)
                    cv2.putText(image_for_result, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS[pred_class], 3)

                if CLASSES[pred_class] == "person":
                    personMiddlePoint = (((ptA[0]+ptB[0])/2),((ptA[1]+ptB[1])/2))
                    personXrightlimit = ptB[0]
                    personXleftlimit = ptA[0]
                    personXhalfSide = abs(personMiddlePoint[0] - personXrightlimit)
                    tperson = time.perf_counter()
                    person = 1
                    if tv == 1:
                        if abs(ttv-tperson) < 0.5:
                            if abs(personMiddlePoint[0] - tvMiddlePoint[0]) < (personXhalfSide + tvXhalfSide):
                                print(abs(personMiddlePoint[0] - tvMiddlePoint[0]))
                                print((personXhalfSide + tvXhalfSide))
                                OPEN = True
                        else:
                            OPEN = False
                            tv = 0

                if CLASSES[pred_class] == "tvmonitor":
                    tvMiddlePoint = (((ptA[0]+ptB[0])/2),((ptA[1]+ptB[1])/2))
                    tvXrightlimit = ptB[0]
                    tvXleftlimit = ptA[0]
                    tvXhalfSide = abs(tvMiddlePoint[0] - tvXrightlimit)
                    ttv = time.perf_counter()
                    tv = 1
                    if person == 1:
                        if abs(ttv-tperson) < 0.5:
                            if abs(personMiddlePoint[0] - tvMiddlePoint[0]) < (personXhalfSide + tvXhalfSide):
                                print(abs(personMiddlePoint[0] - tvMiddlePoint[0]))
                                print((personXhalfSide + tvXhalfSide))
                                OPEN = True
                        else:
                            OPEN = False
                            person = 0

                if OPEN == True and CanSwitch == False:
                    print("Opening the door")
                    publish.single("ledStatus", "1", hostname="192.168.0.21")
                    CanSwitch = True

                if OPEN == False and CanSwitch == True:
                    print("Closing the door")
                    publish.single("ledStatus", "0", hostname="192.168.0.21")
                    CanSwitch = False

        # check if we should display the frame on the screen
        # with prediction data (you can achieve faster FPS if you
        # do not output to the screen)
        if args["display"] > 0:
            # display the frame to the screen
            cv2.imshow("Output", image_for_result)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        # update the FPS counter
        fps.update()

    # if "ctrl+c" is pressed in the terminal, break from the loop
    except KeyboardInterrupt:
        break

    # if there's a problem reading a frame, break gracefully
    except AttributeError:
        break

close_ncs_device()
