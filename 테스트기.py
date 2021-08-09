#!/usr/bin/env python
# coding: utf-8



import pyaudio
import wave
import tkinter
from tkinter import *
from tkinter import Button
from keras.models import Sequential, Model, model_from_json, load_model
import matplotlib.pyplot as plt
import keras 
import pickle
import os
import numpy as np
import sys
import warnings
import librosa
import librosa.display
import IPython.display as ipd  # To play sound in the notebook


class Emo:
    def __init__(self):
        self.root = Tk()
        self.root.title("감정 맞추기")
        self.root.geometry("640x500+100+100")
        self.root.resizable(True, True)

        
        self.label=tkinter.Label(self.root, text="\n안녕하세요 저는 당신의 이야기를 들어줄 PUMP 랍니다!\n당신의 이야기를 들어드릴게요.\n", font = (30))
        self.label.pack()
        
        photo=tkinter.PhotoImage(file='robot.png')
        self.label=tkinter.Label(self.root, image=photo)
        self.label.image = photo
        self.label.pack()

        b1 = tkinter.Button(self.root,width=10 ,height=5,text = "5초간\n대화 하기",command = lambda:[self.Rec(),self.Analysis()])
        b1.pack()
        self.label = tkinter.Label(self.root, text = "\n당신의 감정은?", font = (30))
        self.label.pack()



    def Rec(self):

       
      
       
        CHUNK = 1024
        FORMAT = pyaudio.paInt16 #paInt8
        CHANNELS = 2 
        RATE = 44100 #sample rate
        RECORD_SECONDS = 5
        WAVE_OUTPUT_FILENAME = "output10.wav"

        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK) #buffer
        print("* recording")

        frames = []

        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data) # 2 bytes(16 bits) per channel

        print("* done recording")

        stream.stop_stream()
        stream.close()
        p.terminate()
        

        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        


    def Analysis(self):



        # ignore warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
        if not sys.warnoptions:
            warnings.simplefilter("ignore")
            
        with open("labels","rb") as f:
            label = pickle.load(f)
        os.path.abspath(__file__)

        model = load_model("Emotion_Model.h5")
        model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
        X, sampling_rate = librosa.load('output10.wav',res_type='kaiser_fast'
                                        ,duration=2.5
                                        ,sr=44100
                                        ,offset=0.5
                                        )
        sampling_rate = np.array(sampling_rate)

        ipd.Audio('output10.wav')
        json_file = open('model_json.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("Emotion_Model.h5")
        print("Loaded model from disk")
        opt = keras.optimizers.RMSprop(lr=0.00001, decay=1e-6)
        loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


        mfccs = np.mean(librosa.feature.mfcc(y=X,
                                        sr=sampling_rate, 
                                        n_mfcc=13),
                    axis=0)

        mean = np.mean(mfccs, axis=0)
        std = np.std(mfccs,axis=0)
        Test = (mfccs-mean)/std
        Test = np.array(Test)


        Test = np.expand_dims(Test, axis=1)
        Test = np.expand_dims(Test, axis=0)


        pred = np.argmax(loaded_model.predict(Test))




        print(label.classes_[pred])
        

        emotion = []
        if label.classes_[pred] == 'angry':
            emotion = "\n에구머니나 화나셨군요!"

        elif label.classes_[pred] == 'happy':
            emotion = "\n어머 기쁘시군요!"
        elif label.classes_[pred] == 'sad':
            emotion = "\n엉엉ㅠㅠ 슬프시군요"
        elif label.classes_[pred] == 'neutral':
            emotion = "\n오잉 감정의 변화가 없으시군요"
        self.label.config(text = emotion, font = (30))
        self.label.pack()
        
        


    











main = Emo()














# load weights into new model


# the optimiser


