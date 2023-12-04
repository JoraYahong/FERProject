
import pandas as pd
import numpy as np
import sys
from PIL import Image
import os



outdir = os.path.join('D:/PYproject/FER/CKplus/')


ferplus = pd.read_csv('D:/PYproject/FER/fer2013new.csv')
fer2013 = pd.read_csv('D:/PYproject/FER/fer2013.csv')
print('Read fer2013new success!')

if len(fer2013)!=len(ferplus):
    print('fer2013.csv and fer2013new.csv have different rows.')
    print('Please check them!')
    sys.exit()

emotions = {'0':'Neutral',
            '1':'Happy',    
            '2':'Surprise',     
            '3':'Sadness',     
            '4':'Anger',     
            '5':'Disgust',   
            '6':'Fear',
            '7':'Contempt'
            }

os.makedirs(os.path.join(outdir,'Training'))
os.makedirs(os.path.join(outdir,'PublicTest'))
os.makedirs(os.path.join(outdir,'PrivateTest'))

for i in range(8):
    os.makedirs(os.path.join(outdir,'Training',emotions[str(i)]))
    os.makedirs(os.path.join(outdir,'PublicTest',emotions[str(i)]))
    os.makedirs(os.path.join(outdir,'PrivateTest',emotions[str(i)]))

def SaveImage(imgstr, imgpath):
    img = list(map(np.uint8, imgstr.split()))
    img = np.array(img).reshape([48,48])
    img = Image.fromarray(img)
    img.save(imgpath)


votes = np.zeros([len(ferplus),10])
for i in range(votes.shape[0]):
    for j in range(votes.shape[1]):
        votes[i,j] = ferplus.loc[i][j+2]


t = 1+sys.float_info.epsilon  
for i in range(votes.shape[0]):
    for j in range(votes.shape[1]):
        if votes[i,j]<t:
            votes[i,j] = 0.


idx = []
lab = []
img_count = np.zeros(8,dtype = np.int)
#index = []
#exps = []
for i in range(votes.shape[0]):
    tmp = votes[i]
    maxval = max(tmp)
    ind = np.argmax(tmp)
    if maxval>0.5*tmp.sum() and ind <8:
            iname = os.path.join(outdir,fer2013.loc[i][2],emotions[str(ind)],str(i)+'.png')
            img = fer2013.loc[i][1]
            SaveImage(img,iname)
            img_count[ind] += 1
            #index.append(i)
            #exps.append(emotions[str(ind)])
print()
for i in range(8):
    print(emotions[str(i)], 'images:', img_count[i])
print('Total images:', img_count.sum())
print()   
print('Extraction finished!')
