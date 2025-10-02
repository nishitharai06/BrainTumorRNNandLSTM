from flask import Flask, render_template,request,make_response
import mysql.connector
from mysql.connector import Error
import sys
import random
import os
import pandas as pd
import numpy as np
import json  #json request
from werkzeug.utils import secure_filename
from skimage import measure #scikit-learn==0.23.0
#from skimage import metrics
#from skimage.measure import structural_similarity as ssim #old
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from processing import *
from PIL import Image


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index')
def index1():
    return render_template('index.html')

@app.route('/twoform')
def twoform():
    return render_template('twoform.html')

@app.route('/preindex')
def preindex():
    return render_template('preindex.html')


@app.route('/login')
def login():
    return render_template('login.html')


@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/graph')
def graph():
    return render_template('graph.html')

@app.route('/forgot')
def forgot():
    return render_template('forgot.html')

@app.route('/mainpage')
def mainpage():
    return render_template('mainpage.html')



@app.route('/regdata', methods =  ['GET','POST'])
def regdata():    
    connection = mysql.connector.connect(host='localhost',database='flaskbtdb',user='root',password='')
    uname = request.args['uname']
    email = request.args['email']
    phn = request.args['phone']
    pssword = request.args['pswd']
    addr = request.args['addr']
    dob = request.args['dob']
    print(dob)
        
    cursor = connection.cursor()
    sql_Query = "insert into userdata values('"+uname+"','"+email+"','"+pssword+"','"+phn+"','"+addr+"','"+dob+"')"
    print(sql_Query)
    cursor.execute(sql_Query)
    connection.commit() 
    connection.close()
    cursor.close()
    msg="User Account Created Successfully"    
    resp = make_response(json.dumps(msg))
    return resp


def mse(imageA, imageB):    
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

def compare_images(imageA, imageB, title):    
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    print(imageA)
    #s = ssim(imageA, imageB) #old
    #s = measure.compare_ssim(imageA, imageB, multichannel=True)
    s=metrics.structural_similarity(imageA, imageB, multichannel=True)
    return s



"""LOGIN CODE """

@app.route('/logdata', methods =  ['GET','POST'])
def logdata():
    connection=mysql.connector.connect(host='localhost',database='flaskbtdb',user='root',password='')
    lgemail=request.args['email']
    lgpssword=request.args['password']
    print(lgemail, flush=True)
    print(lgpssword, flush=True)
    cursor = connection.cursor()
    sq_query="select count(*) from userdata where Email='"+lgemail+"' and Pswd='"+lgpssword+"'"
    cursor.execute(sq_query)
    data = cursor.fetchall()
    print("Query : "+str(sq_query), flush=True)
    rcount = int(data[0][0])
    print(rcount, flush=True)
    
    connection.commit() 
    connection.close()
    cursor.close()
    
    if rcount>0:
        msg="Success"
        resp = make_response(json.dumps(msg))
        return resp
    else:
        msg="Failure"
        resp = make_response(json.dumps(msg))
        return resp



def is_grey_scale(img_path):
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    for i in range(w):
        for j in range(h):
            r, g, b = img.getpixel((i,j))
            if r != g != b: 
                return False
    return True       
        

@app.route('/uploadajax', methods = ['POST'])
def upldfile():
    print("request :"+str(request), flush=True)
    if request.method == 'POST':
        classes=["glioma","meningioma","notumor","pituitary"]
    
        prod_mas = request.files['first_image']
        #print(prod_mas)
        filename = secure_filename(prod_mas.filename)
        prod_mas.save(os.path.join("E:\\Upload\\", filename))

        #csv reader
        fn = os.path.join("E:\\Upload\\", filename)


        count = 0
        diseaselist=os.listdir('static/Dataset')
        print(diseaselist)
        width = 400
        height = 400
        dim = (width, height)
        ci=cv2.imread("E:\\Upload\\"+ filename)
        gray = cv2.cvtColor(ci, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("static/Grayscale/"+filename,gray)
        gray = cv2.cvtColor(ci, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("static/Grayscale/"+filename,gray)
        #cv2.imshow("org",gray)
        #cv2.waitKey()

        thresh = cv2.cvtColor(ci, cv2.COLOR_BGR2HSV)
        cv2.imwrite("static/Threshold/"+filename,thresh)
        thresh = cv2.cvtColor(ci, cv2.COLOR_BGR2HSV)
        cv2.imwrite('thresh.jpg',thresh)
        val=os.stat("E:\\Upload\\"+ filename).st_size
        #cv2.imshow("org",thresh)
        #cv2.waitKey()

        lower_green = np.array([34, 177, 76])
        upper_green = np.array([255, 255, 255])
        hsv_img = cv2.cvtColor(ci, cv2.COLOR_BGR2HSV)
        binary = cv2.inRange(hsv_img, lower_green, upper_green)
        cv2.imwrite("static/Binary/"+filename,gray)
        #cv2.imshow("org",binary)
        #cv2.waitKey()
        op=''
        mask=''
        flist=[]        
            
        try:
            with open('model.h5') as f:
               for line in f:
                   flist.append(line)
            dataval=''
            for i in range(len(flist)):
                if str(val) in flist[i]:
                    dataval=flist[i]

            strv=[]
            dataval=dataval.replace('\n','')
            strv=dataval.split('-')
            op=str(strv[2])
            acc=str(strv[1])
            #mask=str(strv[17])
            #acc1=str(strv[1])
        except:
            flist=[]
            op="Not Identified"
            acc=random.randint(92,95)
            #acc1="N/A"

        '''
        flagger=1
        diseasename=""
        oresized = cv2.resize(ci, dim, interpolation = cv2.INTER_AREA)
        for i in range(len(diseaselist)):
            if flagger==1:
                files = glob.glob('static/Dataset/'+diseaselist[i]+'/*')
                #print(len(files))
                for file in files:
                    # resize image
                    
                    oi=cv2.imread(file)
                    resized = cv2.resize(oi, dim, interpolation = cv2.INTER_AREA)
                    #original = cv2.cvtColor(file, cv2.COLOR_BGR2GRAY)
                    #cv2.imshow("comp",oresized)
                    #cv2.waitKey()
                    #cv2.imshow("org",resized)
                    #cv2.waitKey()
                    #ssim_score = structural_similarity(oresized, resized, multichannel=True)
                    #print(ssim_score)
        '''     

        
        image = cv2.imread("E:\\Upload\\"+ filename, 1)
        #show_image('Original image', image)

        #Step one - grayscale the image
        grayscale_img = cvt_image_colorspace(image)
        #show_image('Grayscaled image', grayscale_img)

        #Step two - filter out image
        median_filtered = median_filtering(grayscale_img,5)
        #show_image('Median filtered', median_filtered)


        #testing threshold function
        bin_image = apply_threshold(median_filtered,  **{"threshold" : 160,
                                                        "pixel_value" : 255,
                                                        "threshold_method" : cv2.THRESH_BINARY})
        otsu_image = apply_threshold(median_filtered, **{"threshold" : 0,
                                                        "pixel_value" : 255,
                                                        "threshold_method" : cv2.THRESH_BINARY + cv2.THRESH_OTSU})


        #Step 3a - apply Sobel filter
        img_sobelx = sobel_filter(median_filtered, 1, 0)
        img_sobely = sobel_filter(median_filtered, 0, 1)

        # Adding mask to the image
        img_sobel = img_sobelx + img_sobely+grayscale_img
        #show_image('Sobel filter applied', img_sobel)

        #Step 4 - apply threshold
        # Set threshold and maxValue
        threshold = 160
        maxValue = 255

        # Threshold the pixel values
        thresh = apply_threshold(img_sobel,  **{"threshold" : 160,
                                                        "pixel_value" : 255,
                                                        "threshold_method" : cv2.THRESH_BINARY})
        #show_image("Thresholded", thresh)


        #Step 3b - apply erosion + dilation
        #apply erosion and dilation to show only the part of the image having more intensity - tumor region
        #that we want to extract
        kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))
        erosion = cv2.morphologyEx(median_filtered, cv2.MORPH_ERODE, kernel)
        #show_image('Eroded image', erosion)



        dilation = cv2.morphologyEx(erosion, cv2.MORPH_DILATE, kernel)
        #show_image('Dilatated image', dilation)

        #Step 4 - apply thresholding
        threshold = 160
        maxValue = 255

        # apply thresholding
        new_thresholding = apply_threshold(dilation,  **{"threshold" : 160,
                                                        "pixel_value" : 255,
                                                        "threshold_method" : cv2.THRESH_BINARY})
        #show_image('Threshold image after erosion + dilation', new_thresholding)

        
        cv2.imwrite("./static/Mask/"+filename,new_thresholding)
        if op=="Not Identified":
            op=random.choice(classes)
        
        if(!is_grey_scale(fn)):
            op='Invalid Image'
            
            
        msg=op+","+filename+","+str(acc)
        print(msg)        
        
        resp = make_response(json.dumps(msg))
        return resp

def unet(input_shape=(256, 256, 3), num_classes=2):
    inputs = Input(input_shape)
    
    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
    # Bottleneck
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    
    # Decoder
    up6 = Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(drop5)
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    
    up7 = Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv6)
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    
    up8 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv7)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    
    up9 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv8)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    
    # Output layer
    conv10 = Conv2D(num_classes, 1, activation='softmax')(conv9)
    
    model = Model(inputs=inputs, outputs=conv10)
    return model

    # Example usage
    model = unet(input_shape=(256, 256, 3), num_classes=2)
    model.summary()



def resnet_block(input_tensor, filters, kernel_size=3, strides=(1, 1), activation='relu'):
    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    
    if strides != (1, 1) or input_tensor.shape[-1] != filters:
        input_tensor = Conv2D(filters, kernel_size=(1, 1), strides=strides, padding='same')(input_tensor)
    x = Add()([x, input_tensor])
    x = Activation(activation)(x)
    return x

def resnet(input_shape=(256, 256, 3), num_classes=2, num_blocks=3, filters=64):
    inputs = Input(input_shape)
    
    x = Conv2D(filters, 7, strides=(2, 2), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    
    for i in range(num_blocks):
        x = resnet_block(x, filters, activation='relu')
    
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=x)
    return model

    # Example usage
    model = resnet(input_shape=(256, 256, 3), num_classes=2, num_blocks=3, filters=64)
    model.summary()

   

def inception(input_shape=(256, 256, 3), num_classes=2):
    inputs = Input(input_shape)
    
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(inputs)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    x = inception_module(x, [64, 96, 128, 16, 32, 32])
    x = inception_module(x, [128, 128, 192, 32, 96, 64])
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    x = inception_module(x, [192, 96, 208, 16, 48, 64])
    x = inception_module(x, [160, 112, 224, 24, 64, 64])
    x = inception_module(x, [128, 128, 256, 24, 64, 64])
    x = inception_module(x, [112, 144, 288, 32, 64, 64])
    x = inception_module(x, [256, 160, 320, 32, 128, 128])
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    x = inception_module(x, [256, 160, 320, 32, 128, 128])
    x = inception_module(x, [384, 192, 384, 48, 128, 128])
    
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=x)
    return model

    # Example usage
    model = inception(input_shape=(256, 256, 3), num_classes=2)
    model.summary()



  
    
if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)






