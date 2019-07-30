
#code to experiment on the cv2.QRCodeDetector() module of python

import cv2
import numpy as np
import sys
import time
from tkinter import messagebox

def highlight_qr(code_img,points):
    '''
        Function to mark the QR code in the given image
    '''
    for i in range(0,len(points)):
        print("value of i : ",i)
        print("points[{}] : {}".format(i,points[i]))
        cv2.line(code_img,tuple(points[i][0]),tuple(points[(i+1)%4][0]),(0,0,255),5)
    return code_img

if __name__ == "__main__":
    #read the image
    #qr_code_img = cv2.imread('frame.png')
    qr_code_img = cv2.imread('frame_no_www.png')
    cv2.imshow("QR Code",qr_code_img)
    cv2.waitKey(0)
    #create object for QRCodeDetector
    qr_detector = cv2.QRCodeDetector()
    
    #detect the QR code using detect()
    retval,points = qr_detector.detect(qr_code_img)
    print("retval from detect() : ",retval)
    print("points from detect() : ",points, len(points))
    for co_ordinate in points:
        i=1
        print("value of point returned from loop {} is {}".format(i,co_ordinate))
        print("length and type ",len(co_ordinate),type(co_ordinate))
        print("1st val of co_ordinate : ",co_ordinate[0],len(co_ordinate[0]))
        i = i+1
    #highlight_qr(qr_code_img, points)

    #decode the qr code using decode()
    retval2,straight_qrcode = qr_detector.decode(qr_code_img,points)
    #print("retval from decode() : ",retval2)
    #print("straight_qrcode from decode() : ",straight_qrcode)
    #cv2.imshow("straight_qrcode from decode()",straight_qrcode)
    #cv2.waitKey(0)

    #highlight the QR code to user
    detected_qr = highlight_qr(qr_code_img, points)
    cv2.imshow("Marked QR ",detected_qr)
    cv2.waitKey(0)

    print("\nDecoded value from QR Code : ",retval2,"\n")
    messagebox.showinfo("Decoded QR code",retval2)
