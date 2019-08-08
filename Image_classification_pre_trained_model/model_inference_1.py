# code to test model inferencing(using pre-trained models) using pytorch.

from tkinter import messagebox
from torchvision import transforms
from torchvision import models
import torch
from PIL import Image
import cv2

#print all available models
#it prints all classes and the functions that can be used
print(dir(models))

#using alexnet model

#step-I: Loading the model
alexnet = models.alexnet(pretrained=True)
#check out some details of the network’s architecture
#print(alexnet)

#printing functions available in transforms
print("functions available in transforms : \n",dir(transforms))

#step-II: Specify image transformations
transform = transforms.Compose([            #[1]
 transforms.Resize(256),                    #[2]
 transforms.CenterCrop(224),                #[3]
 transforms.ToTensor(),                     #[4]
 transforms.Normalize(                      #[5]
 mean=[0.485, 0.456, 0.406],                #[6]
 std=[0.229, 0.224, 0.225]                  #[7]
 )])

#Step 3: Load the input image and pre-process it
#img = Image.open("dog.jpg")
#image = cv2.imread("dog.jpg",-1)
#img = Image.open("cat.jpg")
img = Image.open("nilanjande2.jpeg")
#image = cv2.imread("cat.jpg",-1)
image = cv2.imread("nilanjande2.jpeg",-1)
cv2.imshow("image",image)
cv2.waitKey(0)
img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)

#Step 4: Model Inference
alexnet.eval() #Put the model in eval mode
out = alexnet(batch_t)
print("="*50)
#print(out)
print("="*50)
print(out.shape)
with open('imagenet_classes.txt') as f:
    classes = [line.strip() for line in f.readlines()]

#print("classes : ",classes)

#Step 5: print the label
value , index = torch.max(out, 1)
print("value, index : ",value,index)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
#print("Percentage : ",percentage)
#[(classes[idx], percentage[idx].item()) for idx in index[0][:5]] 
[print(classes[idx], percentage[idx].item()) for idx in index] 
#print(labels[index[0]], percentage[index[0]].item())

messagebox.showinfo("predicted class",classes[index])

#using resnet model
# First, load the model
resnet = models.resnet101(pretrained=True)
 
# Second, put the network in eval mode
resnet.eval()
 
# Third, carry out model inference
out = resnet(batch_t)
 
# Forth, print the top 5 classes predicted by the model
_, indices = torch.sort(out, descending=True)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
print("="*50)
print("Top 5 models predicted by resnet model")
[print(classes[idx], percentage[idx].item()) for idx in indices[0][:5]]
