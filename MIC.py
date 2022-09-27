# This file contains the code for the multiclass image classification of the dataset
# The dataset used for this code is attached in the repository, i.e., 50 bark texture dataset

# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os

dir_path=r"C:\\Users\\RISHI\\Music\\FELLOWSHIP\\data\\BarkVN-50\\BarkVN-50_mendeley\\"

res = os.listdir(dir_path)
print(res)

with open('C:\\Users\\RISHI\\Music\\FELLOWSHIP\\categories.txt','w') as f:
    for word in res:
        f.write(word+',')

rows=open("C:\\Users\\RISHI\\Music\\FELLOWSHIP\\categories.txt").read().strip().split(',')
classes = [r[r.find(' ')+1:].split(',')[0] for r in rows]

# load the input image from disk
image=cv2.imread("C:\\Users\\RISHI\\Music\\FELLOWSHIP\\data\\BarkVN-50\\BarkVN-50_mendeley\\Acacia\\IMG_6348.JPG")

# our CNN requires fixed spatial dimensions for our input image(s)
# so we need to ensure it is resized to 224x224 pixels while
# performing mean subtraction (104, 117, 123) to normalize the input;
# after executing this command our "blob" now has the shape:
# (1, 3, 224, 224)
blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))

# load our serialized model from disk
print("[INFO] loading model...")
#net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
net=cv2.dnn.readNetFromCaffe("C:\\Users\\RISHI\\Music\\FELLOWSHIP\\deploy.prototxt","C:\\Users\\RISHI\\Music\\FELLOWSHIP\\bvlc_googlenet.caffemodel")

# set the blob as input to the network and perform a forward-pass to
# obtain our output classification
net.setInput(blob)
start = time.time()
preds = net.forward()
end = time.time()
print("[INFO] classification took {:.5} seconds".format(end - start))

# sort the indexes of the probabilities in descending order (higher
# probabilitiy first) and grab the top-5 predictions
idxs = np.argsort(preds[0])[::-1][:5]

# loop over the top-5 predictions and display them
for (i, idx) in enumerate(idxs):
    # draw the top prediction on the input image
    if i == 0:
        text = "Label: {}, {:.2f}%".format(classes[list(idxs).index(idx)],preds[0][list(idxs).index(idx)] * 100)
        cv2.putText(image, text, (5, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
            
# display the predicted label + associated probability to the console
print("[INFO] {}. label: {}, probability: {:.5}".format(i + 1,classes[i], preds[0][i]))
# display the output image
cv2.imshow("Image", image)
cv2.waitKey(0)
