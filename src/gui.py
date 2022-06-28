# importing libraries
import tkinter as tk
from tkinter import Message, Text
import cv2
import os
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import PIL
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from pathlib import Path
import tensorflow as tf
from functions import emotion_detection, get_model,detectImage
from tensorflow.keras.applications import VGG16
from PIL import Image  

window = tk.Tk()
window.title("Face emotion Recognicion")
window.configure(background ='white')
window.grid_rowconfigure(0, weight = 1)
window.grid_columnconfigure(0, weight = 1)
message = tk.Label(
    window, text ="Une application de la reconnaissance des émotions par machine learning",
    bg ="green", fg = "white", width = 52,
    height = 3, font = ('times', 30, 'bold'))
     
message.place(x = 200, y = 20)
def load_model(name='../models/custom_modelNet'):
    new_model = tf.keras.models.load_model(name+'.h5')
    return new_model

g_model = load_model('../models/custom_model1Net')

   
def startcamera():
    # model = load_model()
    emotion_detection(g_model)


def uploadImage():
    global img
    f_types = [('Jpg Files', '*.jpg')]
    filename = filedialog.askopenfilename(filetypes=f_types)
    # img = ImageTk.PhotoImage(file=filename)
    img = PIL.Image.open(filename)
    # opencvImage = cv2.cvtColor(numpy.array(pil_image), cv2.COLOR_RGB2BGR)
    img = np.array(img)
    frame = detectImage(g_model, img)
    img = Image.fromarray(frame)
    img = ImageTk.PhotoImage(img)
    b2 =tk.Button(window,image=img) # using Button 
    b2.place(x=500,y=200)
    # b2.grid(row=3,column=2)


Y=700
uploadImg = tk.Button(window, text ="Upload Image",
command = uploadImage, fg ="white", bg ="green",
width = 20, height = 3, activebackground = "Red",
font =('times', 15, ' bold '))
uploadImg.place(x = 200, y = Y)

camera = tk.Button(window, text ="Camera",
command = startcamera, fg ="white", bg ="green",
width = 20, height = 3, activebackground = "Red",
font =('times', 15, ' bold '))
camera.place(x = 500, y = Y)
 
quit = tk.Button(window, text ="Quit",
command = window.destroy, fg ="white", bg ="green",
width = 20, height = 3, activebackground = "Red",
font =('times', 15, ' bold '))
quit.place(x = 1000, y = Y)
window.mainloop()