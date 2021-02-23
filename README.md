# Face_Liveness_Detection

## Requirements
All the packages installed on a conda environment
* opencv-contrib-python
* Pillow
* scikit-image
* scikit-learn
* imutils
* matplotlib
* numpy
* tensorflow-gpu(GPU)
* tensorflow(CPU)


## File/Folder Description

__face_eye_detector__: Consists of pretrained deep learning face detector and haarcascade eye detector of OpenCV


__output__: Consists of serialized Keras model, label encoder and a final plot of built model


__plots__: Consists of images/plots showing the training history of multiple experiments and improvements before the final plot.


__test__: Consists of images to evaluate/test from disk.


__gather_examples.py__: This script grabs face ROIs from input video files and helps to create a deep learning face liveness detaset.

`python gather_examples.py --input videos/real.mp4 --output dataset/real --detector face_eye_detector --skip 10`

`python gather_examples.py --input videos/fake.mp4 --output dataset/fake --detector face_eye_detector --skip 25`


__livenessnet.py__: Defined CNN model(VGGNet)


__train.py__: Script to train the model.

`python train.py --dataset dataset --model output/liveness.model --le output/le.pickle --plot output/plot.png`


__predict.py__: Script to test the unseen images from the disk.

`python predict.py --image test/fake.jpg --model output/liveness.model --le output/le.pickle --width 32 --height 32`


__liveness_demo.py__: Script to conduct face liveness detection through webcam in real time.

`python liveness_demo.py --model output/liveness.model --le output/le.pickle --detector face_eye_detector`


## Steps to use in a real time
1 Clone the complete directory

2 Run `pip install -r requirements.txt`

3 Run `python liveness_demo.py --model output/liveness.model --le output/le.pickle --detector face_eye_detector`

Press 'q' to close window and perform cleanup
