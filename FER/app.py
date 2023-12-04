import base64
import time
from myEncoder import MyEncoder
from flask import Flask, request, make_response, send_from_directory, json, jsonify
from flask_cors import *
from utils import get_img_name, result_util, mysql_PooledDB
import threading
import os
import constant
import cv2 as cv
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys
import hashlib
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

import pymysql
from flask import request, Response
import traceback

app = Flask(__name__)
CORS(app, supports_credentials=True)
R = result_util()

db = pymysql.connect(host='localhost',
                     port=3306,
                     user='root',
                     password='root',
                     database='fersys'
                     )

cursor = db.cursor()

lock = threading.Lock()

model_filename = 'D:/MasterProject/PYproject/FER/model/ckplus87375.hdf5'
K.clear_session()
model = load_model(model_filename)
pre = np.expand_dims(np.zeros((48, 48, 3)), axis=0)
model.predict(pre)
print("model loaded!")
original_filesrc = ''
CASCADE_CLASSIFIER_PATH = constant.CASCADE_CLASSIFIER_FILE.replace('\\', '/')

face_detect = cv.CascadeClassifier(CASCADE_CLASSIFIER_PATH)
names = ['Disgust', 'Anger', 'Happy', 'Neutral', 'Surprise', 'Fear', 'Sadness']



def Response_headers(content):
    resp = Response(content)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

def spw(word):
    md5=hashlib.md5()
    md5.update(word.encode())
    return md5.hexdigest()

@app.route('/')
def index():

    return 'home'


@app.route('/loginMessage')
def show():

    return ', choose a mode to continue.'


@app.route('/admin/uploadList', methods=['POST'])
def upload_file_list():

    lock.acquire()
    file = request.form['file']
    #print(type(file))

    if file is None:

        return "No file uploaded."

    filena = request.form['filename']
    #print(filena)
    # filename = get_img_name(file_name=filena)
    filepath = os.path.join(constant.DETECT_UPLOAD_PATH, filena).replace('\\', '/')
    upload_path = os.path.join('/' + constant.RELATIVE_DETECT_UPLOAD_PATH, filena).replace('\\', '/')
    person_name = 'admin'
    imgdata = base64.b64decode(file)
    print(file + '!!!!!!!!!!!!!!!!base64!!!!!!!!!!!')
    fh = open(filepath, "wb")
    fh.write(imgdata)
    fh.close()
    sql = "insert into fileinfo(username,filename,uploadpath) values('" + person_name + "','" + filena + "','" + filepath + "')"
    global original_filesrc
    original_filesrc = file
    try:
        db.ping(reconnect=True)
        cursor.execute(sql)
        db.commit()
        print('insert ok')
    except:

        traceback.print_exc()
        db.rollback()
    finally:

        lock.release()
    db.close()
    while (upload_path == None): {}
    faceDetectRes = 200
    return {'filename': filena, 'faceDetectRes': faceDetectRes}


def face_detect_fun(img, imgpath):

    global save_path
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face = face_detect.detectMultiScale(gray)
    for x, y, w, h in face:
        roi = img[y:y + h, x:x + w]
        re_roi = cv.resize(roi, (48, 48))

        filename = imgpath.replace('.png', 'face') + '.png'
        save_path = os.path.join(constant.DISCERN_UPLOAD_PATH, filename).replace('\\', '/')
        cv.imwrite(save_path, re_roi)
        cv.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
    return save_path


def my_predict_classes(predict_data):
    if predict_data.shape[-1] > 1:
        return predict_data.argmax()
    else:
        return (predict_data > 0.5).astype('int32')


def emotion_detect(detect_img):

    gray = cv.cvtColor(detect_img, cv.COLOR_BGR2GRAY)
    img_data = np.array(detect_img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = img_data / 255
    y_predict = model.predict(img_data)
    print(y_predict)
    y_pre = my_predict_classes(y_predict)
    return names[y_pre]



@app.route('/detect', methods=['POST'])
def detect():
    img_path = request.form['detect_file']

    img_path_new = os.path.join(constant.DETECT_UPLOAD_PATH, img_path).replace('\\', '/')
    result_img = cv.imread(img_path_new)
    print(img_path_new)

    filename = img_path.replace('.png', 'result') + '.png'

    emotiondetectpath = face_detect_fun(result_img, filename)
    resultsavepath = os.path.join(constant.DETECT_RESULT_PATH, filename).replace('\\', '/')

    cv.imwrite(resultsavepath, result_img)


    with open(resultsavepath, 'rb') as f:
        img_stream = base64.b64encode(f.read())
        img64url = img_stream.decode()

    facialEmotionImg = cv.imread(emotiondetectpath)
    emotion_result = emotion_detect(facialEmotionImg)

    detecttime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    person_name = request.form['username']
    sql = "insert into ferrecord(username,ferresult,fertime,imgsrc,filename) values('" + person_name + "','" + emotion_result + "','" + detecttime + "','" + original_filesrc + "','" + img_path + "')"
    sql1 = "UPDATE fileinfo SET resultpath='" + resultsavepath + "',discernpath='" + emotiondetectpath + "' where username='" + person_name + "' and filename='" + img_path + "'"
    try:
        db.ping(reconnect=True)
        cursor.execute(sql)
        cursor.execute(sql1)
        db.commit()
        print('insert ok')
    except:

        traceback.print_exc()
        db.rollback()
    db.close()

    print(emotion_result)
    return {'recogResult': emotion_result, 'faceDetectFile': img64url}



@app.route('/login', methods=['POST'])
def getLoginRequest():

    username1 = request.form['username']
    password = request.form['password']
    resultcode = 0

    sql = "select * from users where username='" + username1 + "' and password='" + password + "'"
    try:

        db.ping(reconnect=True)
        cursor.execute(sql)
        results = cursor.fetchall()
        print(results)
        if len(results) == 1:
            resultcode = 200
        else:
            resultcode = 0

        db.commit()
    except:

        traceback.print_exc()
        db.rollback()

    db.close()
    token = spw(username1+password)
    return {'result': resultcode,'token':token}


@app.route('/register', methods=['POST'])
def getRegisterRequest():

    username2 = request.form['username']
    password2 = request.form['password']
    resultcode = 0

    sql = "select * from users where username='" + username2 + "'"
    sqlReg = "insert into users(username,password) values('" + username2 + "','" + password2 + "')"
    try:

        db.ping(reconnect=True)
        cursor.execute(sql)
        results = cursor.fetchall()
        print(results)
        if len(results) == 1:
            resultcode = 400
        else:
            try:
                resultReg = cursor.execute(sqlReg)
                print(resultReg)
            except:

                traceback.print_exc()
                db.rollback()

        db.commit()
    except:

        traceback.print_exc()
        db.rollback()

    db.close()
    return {'result': resultcode}


@app.route('/recordList', methods=['POST'])
def returnList():
    username = request.form['username']
    sql = "select recid,ferresult,fertime,filename,imgsrc from ferrecord where username='" + username + "'"
    try:

        db.ping(reconnect=True)
        cursor.execute(sql)
        ferresult = cursor.fetchall()
    except:

        traceback.print_exc()
        db.rollback()

    db.close()
    payload = []
    for result in ferresult:
        img64url = result[4].decode()
        content = {'id': result[0], 'ferresult': result[1], 'fertime': result[2], 'filename': result[3],
                   'imgsrc': img64url}
        payload.append(content)
        content = {}
    return jsonify(payload)


@app.route('/recordList/delete', methods=['POST'])
def removeRecord():
    id = request.form['id']
    username = request.form['username']
    filename = request.form['filename']
    sql1 = "DELETE FROM ferrecord where recid=" + id
    sql2 = "select uploadpath,resultpath,discernpath from fileinfo where username='" + username + "' and filename='" + filename + "'"
    sql3 = "DELETE FROM fileinfo where username='" + username + "' and filename='" + filename + "'"
    status = 0
    try:

        db.ping(reconnect=True)
        cursor.execute(sql2)
        deletepath = cursor.fetchall()
        os.remove(deletepath[0][0])
        os.remove(deletepath[0][1])
        os.remove(deletepath[0][2])
        status = 100
    except:
        print("Delete Failed!")
    try:
        db.ping(reconnect=True)
        cursor.execute(sql1)
        cursor.execute(sql3)
        db.commit()
        status = 200
    except:

        traceback.print_exc()
        db.rollback()
    db.close()
    return {"status": status}


@app.route('/changepass', methods=['POST'])
def change():
    oldpass = request.form['oldpass']
    newpass = request.form['newpass']
    print(oldpass)
    user = request.form['user']
    sql = "select * from users where username='" + user + "' and password='" + oldpass + "'"
    sql2 = "UPDATE users SET password='" + newpass + "' where username='" + user + "'"
    status = 0
    try:

        db.ping(reconnect=True)
        cursor.execute(sql)
        results = cursor.fetchall()
        print(results)
        if len(results) == 1:
            cursor.execute(sql2)
            db.commit()
            status = 200
    except:

        traceback.print_exc()
        db.rollback()
    db.close()
    return {"status": status}


@app.route('/recordList/search', methods=['POST'])
def searchRecord():
    word = request.form['searchWord']
    username = request.form['username']
    sql = "select recid,ferresult,fertime,filename,imgsrc from ferrecord where username='" + username + "' and ferresult='" + word + "'"
    status = 0
    try:

        db.ping(reconnect=True)
        cursor.execute(sql)
        results = cursor.fetchall()
        print(results)
        status = 200
    except:

        traceback.print_exc()
        db.rollback()
    db.close()
    payload = []
    for result in results:
        img64url = result[4].decode()
        content = {'id': result[0], 'ferresult': result[1], 'fertime': result[2], 'filename': result[3],
                   'imgsrc': img64url}
        payload.append(content)
        content = {}
    return jsonify(payload)


@app.route('/recordList/changeRes', methods=['POST'])
def changeResult():
    id = request.form['id']
    changeres = request.form['changeRes']
    sql = "UPDATE ferrecord SET ferresult='" + changeres + "' where recid=" + id
    status = 0
    try:

        db.ping(reconnect=True)
        cursor.execute(sql)
        db.commit()
        status = 200
    except:

        traceback.print_exc()
        db.rollback()
    db.close()
    return {"status": status}



if __name__ == '__main__':
    app.run()
