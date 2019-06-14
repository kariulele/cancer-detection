#! /bin/python3

import cv2
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
class Image:
    def __init__(self, name, content, is_cancer):
        self.name = name
        self.content = content
        self.is_cancer = is_cancer
        
    def show(self):
        title = "<"+self.name + "> Cancer: " + str(self.is_cancer)
        plt.gca().set_title( title)
        plt.imshow(self.content)

def openPathToNpArray(path):
    return np.array(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))

def CsvToDicoNameCancer(csvpath):
    df = pd.read_csv(csvpath)
    df = df[:-2]
    df = df.rename(columns={'Nom de l image': 'name', 'Melanome ?': 'is_cancer'})
    df['is_cancer'] = df['is_cancer'] == 1.0
    df['name'] = df['name'] + ".jpg"
    dico = df.set_index(['name']).to_dict()['is_cancer']
    return dico

def getAllImages(basePath, cancerDico):
    files = listdir(basePath)
    storage = {}
    for imgName in files:
        content = openPathToNpArray(basePath + imgName)
        is_cancer = cancerDico[imgName]
        storage[imgName] = Image(imgName, content, is_cancer)
    return storage
   
class ImgStorage: # i.e.: storage =  ImgStorage("../datas/train/")
    def __init__(self, imgPath, csvPath):
        isCancerMap = CsvToDicoNameCancer(csvPath)
        self.localMap = getAllImages(imgPath, isCancerMap)
        self.allList = [self.localMap[name] for name in self.localMap]
        self.allCancerList = [elm for elm in self.allList if elm.is_cancer ]
        self.allNoCancerList = [elm for elm in self.allList if not elm.is_cancer ]
    
    def getImgByName(self, name):
        return self.localMap[name]
    
    def size(self):
        return len(self.localList)

    
storage = ImgStorage('../images/', '../gt_img.csv')

