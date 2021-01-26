from django.shortcuts import render
import cv2
import pytesseract
import os
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn import metrics

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Ruthvik\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# Create your views here.
def home(request):
    return render(request,'shashank/home.html',{'name':'shashank'})

def predict(request):
    val = request.POST['name']
    mail = request.POST['email']
    img = request.POST['report']
    img1 = Image.open(os.path.join(r'C:\Users\Ruthvik\Desktop\oc-py\sample\static\images',img))
    
    img1.load()
    background = Image.new("RGB", img1.size,(255,255,255))
    background.paste(img1,mask = img1.split()[3])
   
    l = pytesseract.image_to_string(background).split()
    
    pima = pd.read_csv(r"C:\Users\Ruthvik\Desktop\oc-py\lung21.csv")
    feature_cols = ['PatientId','Age','Gender','Level','CoughingofBlood']

    X = pima.drop(feature_cols,axis=1)
    y = pima.Level # Target 

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0) 

    logreg = LogisticRegression() 
    logreg.fit(X_train,y_train)
    
    while '' in l:
        l.remove('')

    while ' ' in l:
        l.remove(' ')

    attributes = ['AirPollution','Alcoholuse','DustAllergy','OccuPationalHazards','GeneticRisk','chronicLungDisease','BalancedDiet','Obesity','Smoking','PassiveSmoker','ChestPain','Fatigue','WeightLoss'
                ,'ShortnessofBreath','Wheezing','SwallowingDifficulty','ClubbingofFingerNails','FrequentCold,DryCough','Snoring']

    d = {'PatientId':1}
    for i in range(19):
        if attributes[i] in l:
            d[attributes[i]] = l[l.index(attributes[i])+1]
        else:
            d[attributes[i]] = 0

    new = pd.DataFrame([d])
    m = logreg.predict(new)
    
    return render(request,'shashank/predict.html',{'res':m[0],'image':img,'name':val,'age':30,'email':mail})



