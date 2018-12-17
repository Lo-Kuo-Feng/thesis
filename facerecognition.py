
# coding: utf-8

# In[1]:


import os     #使用操作系統相關功能的模塊
import glob
from shutil import copyfile
import numpy as np          #Python進行科學計算的基礎包
import pandas as pd
import matplotlib.pyplot as plt
import cv2    #OpenCV
import dlib   #dlib是一套包含了機器學習、計算機視覺、圖像處理等的函式庫
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img,img_to_array
from keras.models import load_model

def viewversion():
    import sys
    import numpy
    import pandas
    import matplotlib
    import sklearn
    import tensorflow
    import keras
    import cv2   
    import dlib 

    cd = get_ipython().getoutput('conda --version              # !conda -V')
    jn = get_ipython().getoutput('jupyter notebook --version   #!jupyter notebook -V')
    py = get_ipython().getoutput('python --version             # !python -V')

    print("platform          :",sys.platform)
    print("conda             :",cd[0][6:])
    print("jupyter notebook  :",jn[0])
    print("Python            :",py[0][7:13])
    print("{:<18s}: {}" .format("numpy",numpy.__version__))
    print("{:<18s}: {}" .format("pandas",pandas.__version__))
    print("{:<18s}: {}" .format("matplotlib",matplotlib.__version__))
    print("{:<18s}: {}" .format("sklearn",sklearn.__version__))
    print("{:<18s}: {}" .format("tensorflow",tensorflow.__version__))
    print("{:<18s}: {}" .format("keras",keras.__version__))
    print("{:<18s}: {}" .format("cv2",cv2.__version__))
    print("{:<18s}: {}" .format("dlib",dlib.__version__))

def extractface(sample='sample_face', number=-1, film=0, save_format='jpg'):       #擷取人臉的函數，參數為(VideoCapture參數，樣本編號資料夾,樣本數量)
    if not os.path.exists(sample):              #如果不存在sample的資料夾就創建它
        os.mkdir(sample)
    cap = cv2.VideoCapture(film)                #開啟影片檔案，影片路徑，筆電鏡頭打0
    detector = dlib.get_frontal_face_detector() #Dlib的人臉偵測器
    
    n = 0                                       #人臉圖片編號
    while(cap.isOpened()):                      #使用cap.isOpened()，來檢查是否成功初始化，以迴圈從影片檔案讀取影格，並顯示出來
        ret, frame = cap.read()  #第一個參數ret的值為True或False，代表有沒有讀到圖片;第二個參數是frame，是當前截取一幀的圖片。
        face_rects, scores, idx = detector.run(frame, 0)        #偵測人臉
        for i, d in enumerate(face_rects):      #取出所有偵測的結果
            x1 = d.left()
            y1 = d.top()
            x2 = d.right()
            y2 = d.bottom()
            text = "%2.2f(%d)" % (scores[i], idx[i])            #標示分數，方向
            cropped = frame[int(y1):int(y2),int(x1):int(x2)]    #裁剪偵測到的人臉
            cv2.imwrite(os.getcwd()+"\\{}\\{}_{}.{}".format(sample,sample[:-5],n,save_format), cropped)#儲存裁剪到的人臉
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA) #以方框標示偵測的人臉，cv2.LINE_AA為反鋸齒效果
            cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA) #標示分數
            n += 1                              #更新人臉圖片編號
            if n%100 == 0:                      #每擷取100張人臉圖片時顯示一次擷取數量
                print('已擷取%d張人臉圖片'%n) 
            if n == number:
                break
        cv2.imshow("extract face", frame)     #顯示結果
        if cv2.waitKey(1) & 0xFF == ord('q'):   #按Q停止
            break
        elif n == number:
            break
    print('已擷取{}張人臉樣本'.format(n))
    cap.release()           #釋放資源
    cv2.destroyAllWindows() #刪除任何我們建立的窗口
    
def getnamedict(txt='sample_name.txt'):
    try:
        with open(txt,'r') as f:
            name = f.read().split("\n")
            name_dict = {}
            for i in name:
                key, value = i.split(":")
                name_dict[key] = value
        return name_dict, len(name_dict)
    except ValueError:
        print("No sampl")
        return name_dict, 0
    except FileNotFoundError:
        print("No such file or directory: "+txt)
        return None, None 
        
def train_validation_test_split(txt='sample_name.txt'):
    if os.path.exists(txt):
        name_dict, number_of_samples=getnamedict(txt=txt)
        datasets = ['train', 'validation', 'test']
        sample_face = []
        for i in range(number_of_samples):
            sample_face.append('sample%s'%i+'_face/')
        for dataset in datasets:
            if not os.path.exists(dataset):
                os.mkdir(dataset)
            for i in range(number_of_samples):
                if not os.path.exists(os.path.join(dataset,sample_face[i])):
                    os.mkdir(os.path.join(dataset,sample_face[i]))
                
        for i in range(number_of_samples):
            locals()['sample%s'%i] = os.listdir(sample_face[i])            
        for i in range(number_of_samples):
            locals()['sample%s'%i+'_train_validation'], locals()['sample%s'%i+'_test'] =             train_test_split(locals()['sample%s'%i], test_size = 0.2, random_state = 42)
            print('sample%s'%i+'_train_validation:',len(locals()['sample%s'%i+'_train_validation']),
                  '\tsample%s'%i+'_test:',len(locals()['sample%s'%i+'_test']))
        for i in range(number_of_samples):
            locals()['sample%s'%i+'_train'], locals()['sample%s'%i+'_validation'] =             train_test_split(locals()['sample%s'%i+'_train_validation'], test_size = 0.2, random_state = 42)
            print('sample%s'%i+'_train:',len(locals()['sample%s'%i+'_train']),
                  '\tsample%s'%i+'_validation:',len(locals()['sample%s'%i+'_validation']))
            
        def copyFileToDst(datafolder, srcfolder, filename):
            for f in filename:
                src = os.path.join(srcfolder, f)
                dst = os.path.join(datafolder, srcfolder, f)
                copyfile(src, dst)
        for dataset in datasets:
            for i in range(number_of_samples):
                copyFileToDst(dataset, sample_face[i], locals()['sample%s_'%i+dataset])
                
def dataAugmentation(numbers=100, dataset='train', save_format='jpg'):
    cwd = os.getcwd()
    DIR = os.listdir(dataset)

    for folder in DIR:
        if not os.path.exists(os.path.join(dataset, folder+'+')):              
            os.mkdir(os.path.join(dataset, folder+'+'))
        save_dir = os.path.join(dataset, folder+'+')
        FOLDER = os.listdir(os.path.join(dataset, folder))
        for file in FOLDER:
            path = os.path.join(dataset, folder, file)
            #定义图片生成器
            data_gen = ImageDataGenerator(rotation_range=40,
                                          width_shift_range=0.2,
                                          height_shift_range=0.2,
                                          horizontal_flip=True,
                                          vertical_flip=True,
                                          fill_mode='nearest',
                                          data_format='channels_last')
            img=load_img(path)
            x = img_to_array(img,data_format="channels_last")   #图片转化成array类型,因flow()接收numpy数组为参数
            x=x.reshape((1,) + x.shape)     #要求为4维
            #使用for循环迭代,生成图片
            i = 1
            for batch in data_gen.flow(x,batch_size=1,
                                       save_to_dir=save_dir,
                                       save_prefix='face',
                                       save_format=save_format):
                i += 1
                if i>numbers:
                    break
            os.chdir(os.getcwd()+r'\{}\{}+'.format(dataset,folder))#切換資料夾
            #自動化檔案批次重新命名
            allfiles = glob.glob('*.'+save_format)
            for afile in allfiles:
                os.rename(afile, 'f_'+ afile)
            allfiles = glob.glob('*.'+save_format)
            count=1
            for afile in allfiles:
                new_filename = folder+'_'+str(count) + '.'+save_format
                os.rename(afile, new_filename)
                count += 1
                
            os.chdir(cwd)#切換資料夾
            print('{}已增加了{}筆資料'.format(folder, count-1))
        print('{}共增加了{}筆資料'.format(folder, count-1))
    
def show_acc_history(train_acc='acc',validation_acc='val_acc', history=None):
    try:
        plt.plot(history.history[train_acc])
        plt.plot(history.history[validation_acc])
        plt.title('Train History')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
    except NameError:
        print("name 'history' is not defined")
    
def show_loss_history(train_loss='loss',validation_loss='val_loss', history=None):
    try:
        plt.plot(history.history[train_loss])
        plt.plot(history.history[validation_loss])
        plt.title('Train History')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()     
    except NameError:
            print("name 'history' is not defined")
                
def plot_images_labels_prediction(images,labels,prediction,idx,num=10,txt='sample_name.txt'):
    def getnamedict(txt=txt):
        try:
            with open(txt,'r') as f:
                name = f.read().split("\n")
                name_dict = {}
                for i in name:
                    key, value = i.split(":")
                    name_dict[key] = value
            return name_dict, len(name_dict)
        except ValueError:
            print("No sampl")
            return name_dict, 0
        except FileNotFoundError:
            print("No such file or directory: "+txt)
            return None, None 
    name_dict, number_of_samples = getnamedict()
    
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num>25: num=25 
    for i in range(0, num):
        ax=plt.subplot(5,5, 1+i)
        ax.imshow(images[idx], cmap='binary')
        title= "label=" + name_dict['sample'+str(labels[idx])]
        if len(prediction)>0:
            title+=",predict="+name_dict['sample'+str(prediction[idx])] 
            
        ax.set_title(title,fontsize=10) 
        ax.set_xticks([]);ax.set_yticks([])        
        idx+=1 
    plt.show()
                
def facerecognition(threshold=0.7, film=0, SaveModel='facerecognition.hd5', txt='sample_name.txt'):  
    def getnamedict(txt=txt):
        try:
            with open(txt,'r') as f:
                name = f.read().split("\n")
                name_dict = {}
                for i in name:
                    key, value = i.split(":")
                    name_dict[key] = value
            return name_dict, len(name_dict)
        except ValueError:
            print("No sampl")
            return name_dict, 0
        except FileNotFoundError:
            print("No such file or directory: "+txt)
            return None, None 
    name_dict, number_of_samples = getnamedict()
    if os.path.exists('SaveModel/'+SaveModel):
        model = load_model('SaveModel/'+SaveModel)
        cap = cv2.VideoCapture(film)                                #開啟影片檔案
        detector = dlib.get_frontal_face_detector()              #Dlib的人臉偵測器

        while(cap.isOpened()):       #使用cap.isOpened()，來檢查是否成功初始化，以迴圈從影片檔案讀取影格，並顯示出來
            ret, frame = cap.read()  #第一個參數ret的值為True或False，代表有沒有讀到圖片;第二個參數是frame，是當前截取一幀的圖片。
            face_rects, scores, idx = detector.run(frame, 0)     #偵測人臉
            for i, d in enumerate(face_rects):                   #取出所有偵測的結果
                x1 = d.left()
                y1 = d.top()
                x2 = d.right()
                y2 = d.bottom()
                cropped = frame[int(y1):int(y2),int(x1):int(x2)] #裁剪偵測到的人臉
                image=cv2.resize(cropped,(64, 64),interpolation=cv2.INTER_CUBIC) #將人臉圖片大小調整為(64, 64)
                image = np.expand_dims(image, axis = 0)/255          #增加一個維度
                label = model.predict_classes(image)[0]    #預測類別
                proba = model.predict(image)[0][label]    #預測機率
                name = name_dict['sample'+str(label)]       #利用字典找姓名
                if proba>threshold:
                    text = name+'({}%)'.format(proba)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA) #以方框標示偵測的人臉，cv2.LINE_AA為反鋸齒效果
                    cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)  #標示姓名
                else:
                    text = 'Unlabeled'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4, cv2.LINE_AA) #以方框標示偵測的人臉，cv2.LINE_AA為反鋸齒效果
                    cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)  #標示姓名
            cv2.imshow("face recognition", frame)                  #顯示結果
            if cv2.waitKey(1) & 0xFF == ord('q'):                #按Q停止
                break

        cap.release()                                            #釋放資源
        cv2.destroyAllWindows()                                  #刪除任何我們建立的窗口
    else:
        print("No such file or directory: "+SaveModel)

