from tabnanny import check
from matplotlib.image import imread
import numpy as np
from numpy import imag
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread,imshow
from skimage.transform import resize
import xlsxwriter
import mss
import mss.tools
from keras.models import load_model
from tkinter import *
import tkinter as tk
import win32gui
from PIL import ImageGrab, Image
from operator import indexOf


class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0
        # Creating elements
        self.canvas = tk.Canvas(self, width=300, height=300, bg = "white", cursor="cross")
        self.label = tk.Label(self, text="Draw the letter(A or J) for now you want to recognize with genetics algorithms \n then press the take screen button to save the image \n then close this window", font=("Helvetica", 20))
        self.label1 = tk.Label(self, text="please first maximize this window before captureing the image:", font=("Helvetica", 30))
        self.takeScreen_btn = tk.Button(self, text = "takeScreen", command = self.takeScreen)   
        self.button_clear = tk.Button(self, text = "Clear", command = self.clear_all)
        # Grid structure
        self.canvas.grid(row=6, column=0, pady=2, sticky=W, )
        self.label.grid(row=6, column=1,pady=2, padx=2)
        self.takeScreen_btn.grid(row=8, column=1, pady=2, padx=2)
        self.button_clear.grid(row=8, column=0, pady=2)
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label1.grid(row=0, column=1,pady=2, padx=2)
        #self.canvas.bind("<Motion>", self.start_pos)
        self.canvas.bind("<B1-Motion>", self.draw_lines)
    def clear_all(self):
        self.canvas.delete("all")
    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r=8
        self.canvas.create_oval(self.x-r, self.y-r, self.x + r, self.y + r, fill='black')
    #this function will take screen shot of the written image
    def takeScreen(self):
        with mss.mss() as sct:
            # The screen part to capture
            monitor = {"top": 25, "left": 0, "width": 300, "height": 300}
            output = "checkImage.png".format(**monitor)
            # Grab the data
            sct_img = sct.grab(monitor)
            # Save to the picture file
            mss.tools.to_png(sct_img.rgb, sct_img.size, output=output)
            print(output)


def fuction_to_capture_and_store_the_check_image():
    app = App()
    mainloop()


def function_to_extract_the_features_of_the_initial_population_for_J_dataset():
    # Read the file.
    data = pd.read_excel("character_j.xlsx")
    # See which headers are available.
    print("-------------------------Column headings----------------------------------------")
    print(list(data))
    print("-----------------------------------------------------------------")
    #making array of the image's paths in the dataset
    images=data['image path']
    #definig list of images's graphs
    list_of_images_graphs=[]
    print("--------------------------extracting the verteces of each image in the dataset seperatly---------------------------------------")
    #now i have the dataset and i can loop through it as each image[i] corresponding to the i th image in the dataset
    for i in range(54):
        print("extracting the features grapph for image number:", i," in the dataset")
        readed_image=imread(images[i])
        #the comming code shoud be here::::::::::::::
        #defining array with the same size of image but 2d so i can store the mean of 
        #red ,green,blue valus corresponding to each pexil in the feature matrix
        features_matrix=np.zeros((readed_image.shape[0],readed_image.shape[1]))
        #print("---------------------------------------------------the image shape is---------------------------------------------------")
        #looping throw the original image pixel chanels and calculating the mean values and store them in the coresponding 2d array(feature matrix) 
        for i in range(0,readed_image.shape[0]):
            for j in range(0,readed_image.shape[1]):
                features_matrix[i][j] = ((int(readed_image[i,j,0]) + int(readed_image[i,j,1]) + int(readed_image[i,j,2]))/3)
        #now after the above loop the feature matrix is represnting the image in 2d array
        #print("----------------------------------------resizing the features matrix to be not comlexed--------------------------------------------------------------")
        #resizing the array is just for the sake of reduce the effort althogh it reduce the accuracy but for speeding the algorithm
        resized_image=resize(features_matrix,(30,30))
        #print("-----------------------------------------------making the grapg of the character-------------------------------------------------------")

        #making graph of the dark vertices in the image
        garaph_of_the_current_image=[]
        for i in range(0,resized_image.shape[0]):
            for j in range(0,resized_image.shape[1]):
                if(resized_image[i][j]<=150):
                    garaph_of_the_current_image.append((i,j))
        #adding the graph of the curent image in the list 
        list_of_images_graphs.append(garaph_of_the_current_image)    
        #print("----------------------------------------------the list of garaphs vertices :--------------------------------------------------------")
    return list_of_images_graphs
  

def function_to_extract_the_features_of_the_initial_population_For_a_dataset():
    # Read the file.
    data = pd.read_excel("character_a.xlsx")
    # See which headers are available.
    print("-------------------------Column headings----------------------------------------")
    print(list(data))
    print("-----------------------------------------------------------------")
    #making array of the image's paths in the dataset
    images=data['image path']
    #definig list of images's graphs
    list_of_images_graphs=[]
    print("--------------------------extracting the verteces of each image in the dataset seperatly---------------------------------------")
    #now i have the dataset and i can loop through it as each image[i] corresponding to the i th image in the dataset
    for i in range(54):
        print("extracting the features grapph for image number:", i," in the dataset")
        readed_image=imread(images[i])
        #the comming code shoud be here::::::::::::::
        #defining array with the same size of image but 2d so i can store the mean of 
        #red ,green,blue valus corresponding to each pexil in the feature matrix
        features_matrix=np.zeros((readed_image.shape[0],readed_image.shape[1]))
        #print("---------------------------------------------------the image shape is---------------------------------------------------")
        #looping throw the original image pixel chanels and calculating the mean values and store them in the coresponding 2d array(feature matrix) 
        for i in range(0,readed_image.shape[0]):
            for j in range(0,readed_image.shape[1]):
                features_matrix[i][j] = ((int(readed_image[i,j,0]) + int(readed_image[i,j,1]) + int(readed_image[i,j,2]))/3)
        #now after the above loop the feature matrix is represnting the image in 2d array
        #print("----------------------------------------resizing the features matrix to be not comlexed--------------------------------------------------------------")
        #resizing the array is just for the sake of reduce the effort althogh it reduce the accuracy but for speeding the algorithm
        resized_image=resize(features_matrix,(30,30))
        #print("-----------------------------------------------making the grapg of the character-------------------------------------------------------")

        #making graph of the dark vertices in the image
        garaph_of_the_current_image=[]
        for i in range(0,resized_image.shape[0]):
            for j in range(0,resized_image.shape[1]):
                if(resized_image[i][j]<=150):
                    garaph_of_the_current_image.append((i,j))
        #adding the graph of the curent image in the list 
        list_of_images_graphs.append(garaph_of_the_current_image)    
        #print("----------------------------------------------the list of garaphs vertices :--------------------------------------------------------")
    return list_of_images_graphs
    
#this bellow code was for storing the list of graphs verticec in excel sheet so  i dont have to calculate it each time i use the algorithm
##but when i tried to use the data set its readed as column not rows 
#df = pd.DataFrame(list_of_images_graphs)
# writer = pd.ExcelWriter('list of vertices for each image in the dataset.xlsx', engine='xlsxwriter')
# df.to_excel(writer, sheet_name='intial_population', index=False)
# writer.save()
    
def function_to_extract_the_black_pixels_of_check_image():
    readed_image=imread("checkImage.png",as_gray=False)

    #defining array with the same size of image but 2d so i can store the mean of 
    #red ,green,blue valus corresponding to each pexil in the feature matrix
    features_matrix=np.zeros((readed_image.shape[0],readed_image.shape[1]))
    #print("---------------------------------------------------the image shape is---------------------------------------------------")
    #looping throw the original image pixel chanels and calculating the mean values and store them in the coresponding 2d array(feature matrix) 
    for i in range(0,readed_image.shape[0]):
        for j in range(0,readed_image.shape[1]):
            features_matrix[i][j] = ((int(readed_image[i,j,0]) + int(readed_image[i,j,1]) + int(readed_image[i,j,2]))/3)
    #now after the above loop the feature matrix is represnting the image in 2d array
    #print("----------------------------------------resizing the features matrix to be not comlexed--------------------------------------------------------------")

    #resizing the array is just for the sake of reduce the effort althogh it reduce the accuracy but for speeding the algorithm
    resized_image=resize(features_matrix,(30,30))
    #uncomment the code below to plot the image after resizing
    # plt.imshow(resized_image)
    #plt.show()
    #print("-----------------------------------------------making the grapg of the character-------------------------------------------------------")
    
    #making graph of the dark vertices in the image
    garaph_of_the_current_image=[]
    for i in range(0,resized_image.shape[0]):
        for j in range(0,resized_image.shape[1]):
            if(resized_image[i][j]<=150):
                garaph_of_the_current_image.append((i,j))
    #adding the graph of the curent image in the list 
    return garaph_of_the_current_image

#this function caculate the fitness for between two images
def fitness(pic1,pic2):
    resultt=[]
    small_list=len(pic1)<len(pic2) and pic1 or pic2
    big_list=len(pic1)>len(pic2) and pic1 or pic2
    for i in range(0, len(small_list)):
        resultt.append((abs(pic1[i][0]-pic2[i][0]),abs(pic1[i][1]-pic2[i][1])))
    temp=big_list[len(small_list):]
    resultt.extend(temp)
    sum=0
    for i in range(0, len(resultt)):
        sum+=resultt[i][0]+resultt[i][0]
    return(sum)
#this function calculate the fitness for all the dataset
def clculate_fitness_of_checkImageAndTheinitialPopulation(initialPopulation,checkImage):
    resul=[]
    for i in range(len(initialPopulation)):
        temp=initialPopulation[i]
        resul.append(fitness(temp[:],checkImage))
    return(resul)

#function that take the fitness values and return the index of the n maximum
def TheNFitestIndexOfCromosomes(fitnessArray,numberOfTheNeededCromosomes):
    temp=fitnessArray[:]
    fitestN=[]
    fitestIndexs=[]
    n=0
    #getting the maximum n values
    for i in range(numberOfTheNeededCromosomes):
        a=min(temp)
        fitestN.append(a)
        temp.pop(indexOf(temp,a))
    #geting the indexes for the minimum N values
    for i in range(numberOfTheNeededCromosomes):
        index=indexOf(fitnessArray,fitestN[i])
        fitestIndexs.append(index)
    return(fitestIndexs)

#this function will take the array which contain the n fitest chromosome and will return 
#thier array of genes this function used in the crossover function
def convertTheIndexToCromosom(arrayOfIndexes,initialPopulation):
    ar=[]
    for i in range(len(arrayOfIndexes)):
        ar.append(initialPopulation[arrayOfIndexes[i]])
    return(ar)

#this function calculate the cross over point to be used in the crossOver Function
def crossPoint(paran1,paran2):
    small_list=len(paran1)<len(paran2) and paran1 or paran2
    return(len(small_list)//2)

    #this function performs crossover 
    # bettween two arrays paran1,paran2 around the cross point crospoint and its used by crossover function
def crossoverForTwoLists(paran1,paran2):
    crosPoint=crossPoint(paran1,paran2)
    tempx=paran1[:]
    temp=paran1[0:crosPoint]
    temp1=paran2[crosPoint:]
    temp.extend(temp1)
    arr=temp[:]
    temp=[]
    temp1=[]
    temp=paran2[0:crosPoint]
    temp1=tempx[crosPoint:]
    temp.extend(temp1)
    paran2=temp[:]
    return(paran1,paran2)

#this function will perform the crossover between the list of fitest cromosome and it
#  take the list of indeces of the fitest 
#and return list of fitest cromosomes and the offsprings
def crossOver(listOfNFitestIndex,initialPopulation):
    listOfTheFitestChromosomes=convertTheIndexToCromosom(listOfNFitestIndex,initialPopulation)
    listOfOfsprings=[]
    for i in range(3):
        ofs1,ofs2=crossoverForTwoLists(listOfTheFitestChromosomes[i],listOfTheFitestChromosomes[i+1])
        listOfOfsprings.append(ofs1)
        listOfOfsprings.append(ofs2)
    listOfTheFitestChromosomes.extend(listOfOfsprings)
    return(listOfTheFitestChromosomes)
        

print("---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
print("-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
print("-------------------------------------------------the start of excution-------------------------------------------------------------------------------")
#print("-------------------------------------------------captureing the check image-------------------------------------------------------------------------------")
fuction_to_capture_and_store_the_check_image()
#print("-------------------------------------------------printing the initial population list of list of features-------------------------------------------------------------------------------")
initialPopulationForA= function_to_extract_the_features_of_the_initial_population_For_a_dataset()
print("--------------------------------------------------------------------------------------------------------------------------------")
print("--------------------------------------------------------the check image features vector------------------------------------------------------------------------")
print("---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
print("-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
checkImage=function_to_extract_the_black_pixels_of_check_image()
#print("check image is :-\n",checkImage)
print("---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
print("---------------------------------------------------------the fitness values for a are----------------------------------------------------------------------------------------------------------------------------")
fitListForA=clculate_fitness_of_checkImageAndTheinitialPopulation(initialPopulationForA,checkImage)
theFitestIthsForA=TheNFitestIndexOfCromosomes(fitListForA,5)
wasBEFORETHECROSSOVERForA=convertTheIndexToCromosom(theFitestIthsForA,fitListForA)
print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
print("the fitness was :\n",wasBEFORETHECROSSOVERForA)
print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
listOfNewPopulationForA=crossOver(theFitestIthsForA,initialPopulationForA)
newFitListForA=clculate_fitness_of_checkImageAndTheinitialPopulation(listOfNewPopulationForA,checkImage)
print("the fitness is now:\n",newFitListForA)
print("---------------------------------------------the fitness of being A is -----------------------------------------------------------------------------------------")
probabilityOfBiengA=1/min(newFitListForA)
print(probabilityOfBiengA)


#print("-------------------------------------------------printing the initial population list of list of features-------------------------------------------------------------------------------")
initialPopulationForJDataSet= function_to_extract_the_features_of_the_initial_population_for_J_dataset()
print("---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
print("---------------------------------------------------------the fitness values are for J letter----------------------------------------------------------------------------------------------------------------------------")
fitListForJ=clculate_fitness_of_checkImageAndTheinitialPopulation(initialPopulationForJDataSet,checkImage)
theFitestIthsForJ=TheNFitestIndexOfCromosomes(fitListForJ,5)
wasBEFORETHECROSSOVErForJ=convertTheIndexToCromosom(theFitestIthsForJ,fitListForJ)
print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
print("the fitness for j was :\n",wasBEFORETHECROSSOVErForJ)
print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
listOfNewPopulationForj=crossOver(theFitestIthsForJ,initialPopulationForJDataSet)
newFitListForJ=clculate_fitness_of_checkImageAndTheinitialPopulation(listOfNewPopulationForj,checkImage)
print("the fitness for j is now:\n",newFitListForJ)
print("---------------------------------------------the fitness of being j is -----------------------------------------------------------------------------------------")
probabilityOfBiengj=1/min(newFitListForJ)
print(probabilityOfBiengj)
"""a_file=open("ara.txt","w")
for row in initialPopulation:
    np.savetxt(a_file,row)
a_file.close()
"""
if (probabilityOfBiengA>probabilityOfBiengj):
    print("the handwrtten letter was :- A")
else:
    print("the handwrtten letter was :- j")

