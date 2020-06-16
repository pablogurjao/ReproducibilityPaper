# USAGE
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import paho.mqtt.publish as publish
import numpy as np
import argparse
import imutils
import time
import cv2

global fps, vs
# Used Variables

# initialize the list of class labels our network was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ("background", "aeroplane", "bicycle", "bird",
    "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor")
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

PREPROCESS_DIMS = (300, 300)

def init_params():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--prototxt", required=True,
		help="path to Caffe 'deploy' prototxt file")
	ap.add_argument("-m", "--model", required=True,
		help="path to Caffe pre-trained model")
	ap.add_argument("-c", "--confidence", type=float, default=0.2,
		help="minimum probability to filter weak detections")
	args = vars(ap.parse_args())
	return args["prototxt"],args["model"],args["confidence"]

def preprocess_image(input_image):
    # preprocess the image
    preprocessed = cv2.dnn.blobFromImage(cv2.resize(input_image, PREPROCESS_DIMS),
		0.007843, PREPROCESS_DIMS, 127.5)

    # return the image to the calling function
    return preprocessed

def detect_and_predict(input_image, net):
	global vs
	blob = preprocess_image(input_image)

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()
	return detections

def main():
	global fps, vs
	tv = 0
	person = 0
	CanSwitch = False
	OPEN = False
	file_path,model,conf = init_params()
	# load our serialized model from disk
	print("[INFO] loading model...")
	net = cv2.dnn.readNetFromCaffe(file_path, model)

	# initialize the video stream, allow the cammera sensor to warmup,
	# and initialize the FPS counter
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)
	fps = FPS().start()

	# loop over the frames from the video stream
	while True:
		# grab the frame from the threaded video stream and resize it
		# to have a maximum width of 400 pixels
		frame = vs.read()
		frame = imutils.resize(frame, 400)

		# grab the frame dimensions and convert it to a blob
		(h, w) = frame.shape[:2]
		
		detections = detect_and_predict(frame, net)

		# loop over the detections
		for i in np.arange(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with
			# the prediction
			confidence = detections[0, 0, i, 2]

			# filter out weak detections by ensuring the `confidence` is
			# greater than the minimum confidence
			if confidence > conf:
				# extract the index of the class label from the
				# `detections`, then compute the (x, y)-coordinates of
				# the bounding box for the object
				idx = int(detections[0, 0, i, 1])
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# draw the prediction on the frame
				label = "{}: {:.2f}%".format(CLASSES[idx],
					confidence * 100)
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					COLORS[idx], 2)
				y = startY - 15 if startY - 15 > 15 else startY + 15
				cv2.putText(frame, label, (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
				
				if CLASSES[idx] == "person":
					personMiddlePoint = (((startX+endX)/2),((startY+endY)/2))
					personXrightlimit = endX
					personXleftlimit = startX
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

				if CLASSES[idx] == "bottle":
					tvMiddlePoint = (((startX+endX)/2),((startY+endY)/2))
					tvXrightlimit = endX
					tvXleftlimit = startX
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
					publish.single("ledStatus", "1", hostname="127.0.0.1")
					CanSwitch = True

				if OPEN == False and CanSwitch == True:
					print("Closing the door")
					publish.single("ledStatus", "0", hostname="127.0.0.1")
					CanSwitch = False

		# show the output frame
		cv2.imshow("Frame", frame)
		cv2.waitKey(1)


if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		# update the FPS counter
		fps.update()

		# stop the timer and display FPS information
		fps.stop()
		print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
		print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

		# do a bit of cleanup
		cv2.destroyAllWindows()
		vs.stop()