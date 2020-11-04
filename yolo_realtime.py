from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import numpy as np
import time
import cv2
import os
import imutils

ap = argparse.ArgumentParser()

ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")

ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

#print(net.summary())
#cv2.dnn.
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()
(W, H) = (None, None)
# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=700)
	if W is None or H is None:
		(H, W) = frame.shape[:2]
	# grab the frame dimensions and convert it to a blob
	#blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
	#	0.007843, (300, 300), 127.5)
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	layerOutputs = net.forward(ln)

	boxes = []
	confidences = []
	classIDs = []
	
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			#if CLASSES[classID] in IGNORE:
			#	continue

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > args["confidence"]:
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				box = detection[0:4] * np.array([W, H, W, H])
				#(centerX, centerY, width, height) = box.astype("int")
				(startX, startY, endX, endY) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(startX - (endX / 2))
				y = int(startY - (endY / 2))

				# update our list of bounding box coordinates,
				# confidences, and class IDs
				#label = "{}: {:.2f}%".format(classID,
				#onfidence * 100)
				boxes.append([x, y, int(endX), int(endY)])
				confidences.append(float(confidence))
				classIDs.append(classID)

				
				idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
					args["threshold"])
				
				if len(idxs) > 0:
					# loop over the indexes we are keeping
					for i in idxs.flatten():
						# extract the bounding box coordinates
						(x, y) = (x,y)
						(w, h) = (endX, endY)

						# draw a bounding box rectangle and label on the frame
						color = [int(c) for c in COLORS[classID]]
						cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
						text = "{}: {:.4f}".format(LABELS[classID],
							confidence)
						cv2.putText(frame, text, (x, y - 5),
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
	
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
	# update the FPS counter
	fps.update()
# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()


#run this command : - python yolo_realtime.py --yolo yolo-coco