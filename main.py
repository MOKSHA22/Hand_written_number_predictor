import re
import base64
from flask import Flask, render_template,request
import cv2
import joblib
import numpy  as np
import csv # comma sepereated value lib
from datetime import datetime

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('drawdigits.html')

algorithm=joblib.load('digits_model.svm.sav')

@app.route('/predictdigits/', methods=['GET','POST'])
def predict_digits():
    parseImage(request.get_data()) # send the canvas image to parseImage function
    img=cv2.imread('output.png')
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    resized=cv2.resize(gray,(8,8))
    flatten=np.reshape(resized,(1,64))
    scaled=np.array(flatten/255.0*15,dtype=np.int) # convern to 1 and 0 by dividing by 255, *15 makes them 15 and 0

    result=algorithm.predict(scaled)
    return str(result)
    
    #return str(img.shape)
    # colour image (280, 280, 3)

@app.route('/update',methods=['GET','POST'])
def update():
    result=request.form
    actual_label=int(result['actual'])

    #return str(result)

    now=datetime.now()

    date_time=now.strftime('%m%d%Y %H%M%S')
    img_name=date_time+'.png'

    img=cv2.imread('output.png')
    cv2.imwrite('data/'+img_name,img)
    
    with open('data/dataset.csv','a') as file: # 'a' - append mode
        # use date time to save uniquely
        writer=csv.writer(file)
        writer.writerow([img_name,actual_label])
    return render_template('drawdigits.html')


def crop(im):

    ret,thresh1 = cv2.threshold(im,127,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    i=0
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)         
        if(i==1):
            return thresh1[y:y+h,x:x+w]
        i=i+1

def parseImage(imgData):
    # parse canvas bytes and save as output.png
    imgstr = re.search(b'base64,(.*)', imgData).group(1)
    # decodes the image 
    with open('output.png','wb') as output:
        output.write(base64.decodebytes(imgstr))

if __name__ == '__main__':
    app.run(debug=True)
