# import the necessary packages
from imutils.video import VideoStream
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import pickle
import imutils
import time
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
                help="path to trained model")
ap.add_argument("-l", "--le", type=str, required=True,
                help="path to label encoder")
ap.add_argument("-d", "--detector", type=str, required=True,
                help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load the haarcascades files from the disk
# to detect the eye from face.
eye_cascPath = 'face_eye_detector/haarcascade_eye.xml'
eyeCascade = cv2.CascadeClassifier(eye_cascPath)

# load the liveness detector model and label encoder from disk
print("[INFO] loading liveness detector...")
model = load_model(args["model"])
le = pickle.loads(open(args["le"], "rb").read())

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
	# grab the frame from the threaded video stream
	frame = vs.read()

	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the face and extract the face ROI
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# extract the face ROI and then preprocess it in the exact
			# same manner as our training data
			face = frame[startY:endY, startX:endX]

			# check the eyes on this face
			eyes = eyeCascade.detectMultiScale(face, 1.3, 5)

			if len(eyes) == 2:
				# get into the eyes with its position
				for (a, b, c, d) in eyes:
					# we have to draw the rectangle on the
					# eye
					cv2.rectangle(face, (a, b), (a + c, b + d), (255, 0, 0), thickness=4)

				# preprocess the face ROI in the exact same manner as our training data
				face = cv2.resize(face, (32, 32))
				face = face.astype("float") / 255.0
				face = img_to_array(face)
				face = np.expand_dims(face, axis=0)

				# pass the face ROI through the trained liveness detector
				# model to determine if the face is "real" or "fake"
				preds = model.predict(face)[0]
				j = np.argmax(preds)
				label = le.classes_[j]

				# draw the label and bounding box on the frame
				label = "{}: {:.4f}".format(label, preds[j])
				if le.classes_[j] == 'fake':
					cv2.putText(frame, label, (startX, startY - 10),
								cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
					cv2.rectangle(frame, (startX, startY), (endX, endY),
								  (0, 0, 255), 2)
				else:
					cv2.putText(frame, label, (startX, startY - 10),
								cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
					cv2.rectangle(frame, (startX, startY), (endX, endY),
								  (0, 255, 0), 2)

			else:
				cv2.putText(frame, "Plz, open your eyes for verification", (startX, startY - 10),
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

	# show the output frame and wait for a key press
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()