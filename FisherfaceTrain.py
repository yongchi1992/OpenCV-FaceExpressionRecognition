import cv2
import glob
import random
import numpy as np

emotions = ["neutral","anger","contempt","disgust","fear","happy","sadness","surprise"]
fishface = cv2.face.FisherFaceRecognizer_create()
data = {}

def get_files(emotion):
    files = glob.glob("dataset/%s/*" %emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)]
    prediction = files[int(len(files)*0.8):]
    return training,prediction

def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        training,prediction = get_files(emotion)
        for item in training:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            training_data.append(gray)
            training_labels.append(emotions.index(emotion))

        for item in prediction:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            prediction_data.append(gray)
            prediction_labels.append(emotions.index(emotion))
    return training_data,training_labels,prediction_data,prediction_labels

def run_recognizer():
    training_data,training_labels,prediction_data,prediction_labels=make_sets()

    print "Training fisher face classifier"
    print "size of training set is: ", len(training_labels)," images"

    fishface.train(training_data,np.asarray(training_labels))
    print "Predicting classification set"
    cnt =0 
    correct = 0
    incorrect = 0
    for image in prediction_data:
        pred,conf = fishface.predict(image)
        if pred == prediction_labels[cnt]:
            correct+=1
            cnt+=1
        else:
            cv2.imwrite("difficult/%s_%s_%s.jpg" %(emotions[prediction_labels[cnt]], emotions[pred], cnt), image)
            incorrect+=1
            cnt+=1
    return ((100*correct)/(correct+incorrect))

metascore = []
#for i in range(0,10):
correct = run_recognizer()
print "got ",correct," percent correct!"
metascore.append(correct)
print "\n\nend score: ",np.mean(metascore), "percent correct!"