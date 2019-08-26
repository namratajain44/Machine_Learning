from torchvision import models

from PIL import Image
import matplotlib.pyplot as plt
import torch

import torchvision.transforms as T

import numpy as np

import cv2

fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()
#!wget -nv https://static.independent.co.uk/s3fs-public/thumbnails/image/2018/04/10/19/pinyon-jay-bird.jpg -O bird.png
#img = Image.open('./bird.png')
img = Image.open('./lady.png')
plt.imshow(img); plt.show()
#plt.show(img)

trf = T.Compose([T.Resize(256),
                 T.CenterCrop(224),
                 T.ToTensor(),
                 T.Normalize(mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225])])
#inp_ignore = trf(img)
inp = trf(img).unsqueeze(0)

#print("img with transform without unsqueeze : \n",inp_ignore)

#pass the input through the network
out = fcn(inp)['out']
print ("output shape from model : ",out.shape)
print("output value : \n",out)
#print("sliced output to check the dimensions : ",out[0,21,:,:])
#this gives us the output as [1x21xHxW], which is 21 classes, we need to make it one

#out of 21 values for each pixel position, pick the max one 
om = torch.argmax(out.squeeze(), dim = 0).detach().cpu().numpy()
print("om shape : ",om.shape)
print("value of om : \n",om)

#image could not be shown using cv2.imshow
#cv2.imshow("squeezed output from model",om)
#cv2.waitKey()

# Define the helper function
def decode_segmap(image, nc=21):
  
  label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)
  
  for l in range(0, nc):
    idx = image == l
    r[idx] = label_colors[l, 0]
    g[idx] = label_colors[l, 1]
    b[idx] = label_colors[l, 2]
    
  rgb = np.stack([r, g, b], axis=2)
  return rgb

rgb = decode_segmap(om)
plt.imshow(rgb); plt.show()
