#code to read qr code and barcodes using the pyzbar + opencv library of python

from tkinter import messagebox
import pyzbar.pyzbar as pyzbar
import numpy as np
import cv2
 
def decode(im) : 
  # Find barcodes and QR codes
  decodedObjects = pyzbar.decode(im)
 
  # Print results
  print('decoded objects from pyzbar.decode : ',decodedObjects,type(decodedObjects))
  #decoded objects from pyzbar.decode :  [Decoded(data=b'www.terralogic.com', type='QRCODE', rect=Rect(left=38, top=38, width=223, height=223), polygon=[Point(x=38, y=259), Point(x=261, y=261), Point(x=259, y=38), Point(x=39, y=39)])] <class 'list'>
  for obj in decodedObjects:
    print('Type : ', obj.type)
    print('Data : ', obj.data,type(obj.data),'\n')
    print('string type for data : ',obj.data.decode('ASCII'))
    messagebox.showinfo("Decoded QR",obj.data.decode('ASCII'))
  return decodedObjects
 
 
# Display barcode and QR code location  
def display(im, decodedObjects):
 
  # Loop over all decoded objects
  for decodedObject in decodedObjects: 
    points = decodedObject.polygon
 
    # If the points do not form a quad, find convex hull
    if len(points) > 4 : 
      hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
      hull = list(map(tuple, np.squeeze(hull)))
    else : 
      hull = points;
     
    # Number of points in the convex hull
    n = len(hull)
 
    # Draw the convext hull
    for j in range(0,n):
      cv2.line(im, hull[j], hull[ (j+1) % n], (255,0,0), 3)
 
  # Display results 
  cv2.imshow("Results", im);
  cv2.waitKey(0);
 
   
# Main 
if __name__ == '__main__':
 
  # Read image
  #im = cv2.imread('frame.png')
  #im = cv2.imread('zbar-location.png')
  im = cv2.imread('frame_4.png')
 
  decodedObjects = decode(im)
  #display(im, decodedObjects)
