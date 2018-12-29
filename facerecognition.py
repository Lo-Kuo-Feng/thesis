
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

__version__ = "1.3.1"

def version():
    import sys
    import numpy
    import pandas
    import matplotlib
    import sklearn
    import tensorflow
    import keras
    import cv2   
    import dlib 
    import facerecognition
    
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
    print("{:<18s}: {}" .format("facerecognition",facerecognition.__version__))

def photograph_face(sample_file='photograph_face', sample_name='sample_face', film=0, save_size=64 ,save_format='jpg', show_time=1):
    if not os.path.exists(sample_file):              #如果不存在sample的資料夾就創建它
        os.mkdir(sample_file)
    cap = cv2.VideoCapture(film)                #開啟影片檔案，影片路徑，筆電鏡頭打0
    detector = dlib.get_frontal_face_detector() #Dlib的人臉偵測器
    
    while(cap.isOpened()):       #使用cap.isOpened()，來檢查是否成功初始化，以迴圈從影片檔案讀取影格，並顯示出來
        cv2.namedWindow("photograph face", cv2.WINDOW_NORMAL)
        ret, frame = cap.read()  #第一個參數ret的值為True或False，代表有沒有讀到圖片;第二個參數是frame，是當前截取一幀的圖片。
        frame = cv2.flip(frame,1,dst=None) #水平镜像
        face_rects, scores, idx = detector.run(frame, 0)     #偵測人臉
        big_size = 0
        big_size_idex = 0
        big_size_x1 = 0
        big_size_y1 = 0
        big_size_x2 = 0
        big_size_y2 = 0
        face = False
        for i, d in enumerate(face_rects):                   #取出所有偵測的結果
            x1 = d.left()
            y1 = d.top()
            x2 = d.right()
            y2 = d.bottom()
            height = d.bottom()-d.top()
            width = d.right()-d.left()
            size = height*width
            if  (size > big_size) and x1>0 and y1>0 and x2>0 and y2>0:
                big_size = size
                big_size_idex = i
                big_size_x1 = d.left()
                big_size_y1 = d.top()
                big_size_x2 = d.right()
                big_size_y2 = d.bottom()
                face = True
        if face:
            cropped = frame[int(big_size_y1):int(big_size_y2),int(big_size_x1):int(big_size_x2)] #裁剪偵測到的人臉
            image=cv2.resize(cropped,(save_size, save_size),interpolation=cv2.INTER_CUBIC) #將人臉圖片大小調整為(64, 64)
            text = "%2.2f(%d)" % (scores[i], idx[i])            #標示分數，方向 
            cv2.rectangle(frame, (big_size_x1, big_size_y1), (big_size_x2, big_size_y2), (0, 255, 0), 4, cv2.LINE_AA) #以方框標示偵測的人臉，cv2.LINE_AA為反鋸齒效果
            cv2.putText(frame, text, (big_size_x1, big_size_y1), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)  #標示姓名
            if cv2.waitKey(1) & 0xFF == ord('p'):
                cv2.imwrite(os.getcwd()+"\\{}\\{}.{}".format(sample_file,sample_name,save_format), image)#儲存裁剪到的人臉
                cv2.imshow("photograph face", cropped)                  #顯示結果
                cv2.waitKey(show_time*1000)
        cv2.imshow("photograph face", frame)                  #顯示結果
        if cv2.waitKey(1) & 0xFF == ord('q'):                #按Q停止
            break
    
    cap.release()                                            #釋放資源
    cv2.destroyAllWindows()                                  #刪除任何我們建立的窗口 
    
def extract_face(sample='sample_face', number=-1, film=0, save_format='jpg', view_number=100):#擷取人臉的函數，參數為(VideoCapture參數，樣本編號資料夾,樣本數量)
    if not os.path.exists(sample):              #如果不存在sample的資料夾就創建它
        os.mkdir(sample)
    cap = cv2.VideoCapture(film)                #開啟影片檔案，影片路徑，筆電鏡頭打0
    detector = dlib.get_frontal_face_detector() #Dlib的人臉偵測器
    
    n = 0                                       #人臉圖片編號
    while(cap.isOpened()):                      #使用cap.isOpened()，來檢查是否成功初始化，以迴圈從影片檔案讀取影格，並顯示出來
        cv2.namedWindow("extract face", cv2.WINDOW_NORMAL)
        ret, frame = cap.read()  #第一個參數ret的值為True或False，代表有沒有讀到圖片;第二個參數是frame，是當前截取一幀的圖片。
        frame = cv2.flip(frame,1,dst=None) #水平镜像
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
            if n%view_number == 0:                      #每擷取100張人臉圖片時顯示一次擷取數量
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
    
def get_name_dict(txt='sample_name.txt'):
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
        
def train_validation_test_split(txt='sample_name.txt', tt_split_ratio=0.2, tv_split_ratio=0.2):
    if os.path.exists(txt):
        name_dict, number_of_samples=get_name_dict(txt=txt)
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
        print("--------------------------------------------------------------------------------")
        print("|                                                              |               |")
        print("|              sample_train_validation                         |  sample_test  |")
        print("|                                                              |               |")
        print("--------------------------------------------------------------------------------")
        for i in range(number_of_samples):
            locals()['sample%s'%i+'_train_validation'], locals()['sample%s'%i+'_test'] =             train_test_split(locals()['sample%s'%i], test_size = tt_split_ratio, random_state = 42)
            print('sample%s'%i+'_train_validation:',len(locals()['sample%s'%i+'_train_validation']),
                  '\t\t\t\t\tsample%s'%i+'_test:',len(locals()['sample%s'%i+'_test']))
        print()
        print("--------------------------------------------------------------------------------")
        print("|                                      |                       |               |")
        print("|              sample_train            |  sample_validation    |  sample_test  |")
        print("|                                      |                       |               |")
        print("--------------------------------------------------------------------------------")
        for i in range(number_of_samples):
            locals()['sample%s'%i+'_train'], locals()['sample%s'%i+'_validation'] =             train_test_split(locals()['sample%s'%i+'_train_validation'], test_size = tv_split_ratio, random_state = 42)
            print('sample%s'%i+'_train:',len(locals()['sample%s'%i+'_train']),
                  '\t\t\tsample%s'%i+'_validation:',len(locals()['sample%s'%i+'_validation']),
                  '\tsample%s'%i+'_test:',len(locals()['sample%s'%i+'_test']))
            
        def copy_file_to_dst(datafolder, srcfolder, filename):
            for f in filename:
                src = os.path.join(srcfolder, f)
                dst = os.path.join(datafolder, srcfolder, f)
                copyfile(src, dst)
        for dataset in datasets:
            for i in range(number_of_samples):
                copy_file_to_dst(dataset, sample_face[i], locals()['sample%s_'%i+dataset])
                
def data_augmentation(numbers=100, dataset='train', save_format='jpg', verbose=0):
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
            if verbose==0:
                pass
            elif verbose==1:
                print('{}已增加了{}筆資料'.format(folder, count-1))
        print('{}共增加了{}筆資料'.format(folder, count-1))
    
def show_acc_history(history=None):
    try:
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Train History')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
    except NameError:
        print("name 'history' is not defined")
    
def show_loss_history(history=None):
    try:
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Train History')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()     
    except NameError:
            print("name 'history' is not defined")
        
def evaluation_model(model=None, target_size=64):
 
    from keras.preprocessing import image
    from keras.preprocessing.image import ImageDataGenerator
    from keras.utils import np_utils
    test_datagen = ImageDataGenerator(rescale = 1./255)
    test_set = test_datagen.flow_from_directory('test/', class_mode = 'categorical')
    testset_path = []   #testset_path為所有testset圖片路徑的list
    for i in os.listdir('test'):
        for j in os.listdir('test/'+i):
            testset_path.append('test/'+i+'/'+j)  

    x_test_image = []
    for path in testset_path:
        x_test_image.append(image.img_to_array(image.load_img(path, target_size= (target_size,target_size))))
    x_test_image = np.array(x_test_image)/225

    y_test_label = test_set.classes
    y_Test_OneHot = np_utils.to_categorical(y_test_label)

    test = []
    for x, y, z in zip(x_test_image, y_test_label, y_Test_OneHot):
        test.append([x, y, z])

    x_test_image = []
    y_test_label = []
    y_Test_OneHot = []
    for i in test:
        x_test_image.append(i[0])
        y_test_label.append(i[1])
        y_Test_OneHot.append(i[2])
    x_test_image = np.array(x_test_image)
    y_test_label = np.array(y_test_label)
    y_Test_OneHot = np.array(y_Test_OneHot)

    prediction = model.predict_classes(x_test_image)               #預測
    scores = model.evaluate(x_test_image, y_Test_OneHot, verbose=0)  #評估
    return scores[1]

def crosstab(model=None, txt='sample_name.txt', target_size=64):
    name_dict, number_of_samples = get_name_dict()

    from keras.preprocessing import image
    from keras.preprocessing.image import ImageDataGenerator
    from keras.utils import np_utils
    test_datagen = ImageDataGenerator(rescale = 1./255)
    test_set = test_datagen.flow_from_directory('test/', class_mode = 'categorical')
    testset_path = []   #testset_path為所有testset圖片路徑的list
    for i in os.listdir('test'):
        for j in os.listdir('test/'+i):
            testset_path.append('test/'+i+'/'+j)  

    x_test_image = []
    for path in testset_path:
        x_test_image.append(image.img_to_array(image.load_img(path, target_size= (target_size,target_size))))
    x_test_image = np.array(x_test_image)/225

    y_test_label = test_set.classes
    y_Test_OneHot = np_utils.to_categorical(y_test_label)

    test = []
    for x, y, z in zip(x_test_image, y_test_label, y_Test_OneHot):
        test.append([x, y, z])

    x_test_image = []
    y_test_label = []
    y_Test_OneHot = []
    for i in test:
        x_test_image.append(i[0])
        y_test_label.append(i[1])
        y_Test_OneHot.append(i[2])
    x_test_image = np.array(x_test_image)
    y_test_label = np.array(y_test_label)
    y_Test_OneHot = np.array(y_Test_OneHot)

    prediction = model.predict_classes(x_test_image)               #預測
    y_test_label_names = np.ndarray((len(y_test_label),),dtype=object)
    prediction_names = np.ndarray((len(prediction),),dtype=object)
    for i, j, k in zip(y_test_label, prediction, range(len(y_test_label))):
        y_test_label_names[k] = name_dict['sample'+str(i)]
        prediction_names[k] = name_dict['sample'+str(j)]
    return pd.crosstab(y_test_label_names,prediction_names,rownames=['label'],colnames=['predict'])

def predict(model=None, img=r'sample0_face\sample0_0.jpg', txt='sample_name.txt', target_size=64):  
    name_dict, number_of_samples = get_name_dict()
    
    from keras.preprocessing import image
    test_image = np.expand_dims(image.img_to_array(image.load_img(img, target_size= (target_size,target_size))), 0)/255
    predict = model.predict(test_image)[0]
    predict_proba = model.predict_proba(test_image)[0]
    predict_classes = model.predict_classes(test_image)[0]
    predict_name_proba = model.predict(test_image)[0][model.predict_classes(test_image)[0]]
    predict_name = name_dict['sample'+str(model.predict_classes(test_image)[0])]
    
    import matplotlib.pyplot as plt # plt 用于显示图片
    import matplotlib.image as mpimg # mpimg 用于读取图片
    img_2 = mpimg.imread(img) # 读取和代码处于同一目录下的 lena.png
    # 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
#     img_2.shape #(512, 512, 3)
    plt.imshow(img_2) # 显示图片
    plt.axis('off') # 不显示坐标轴
    plt.show()

    for name, proba in zip(name_dict.values(),predict_proba):
        print("{:<15s}的機率為: {}".format(name, proba))
    print("=========================================")
    print()
    print("預測結果為: {}({}%)".format(predict_name,predict_name_proba))
                
def face_recognition_everyone(model=None, threshold=0.9999999999999999, film=0, txt='sample_name.txt', target_size=64):  
    name_dict, number_of_samples = get_name_dict() 
    cap = cv2.VideoCapture(film)                                #開啟影片檔案
    detector = dlib.get_frontal_face_detector()              #Dlib的人臉偵測器

    while(cap.isOpened()):       #使用cap.isOpened()，來檢查是否成功初始化，以迴圈從影片檔案讀取影格，並顯示出來
        cv2.namedWindow("face recognition everyone", cv2.WINDOW_NORMAL)
        ret, frame = cap.read()  #第一個參數ret的值為True或False，代表有沒有讀到圖片;第二個參數是frame，是當前截取一幀的圖片。
        frame = cv2.flip(frame,1,dst=None) #水平镜像
        face_rects, scores, idx = detector.run(frame, 0)     #偵測人臉
        for i, d in enumerate(face_rects):                   #取出所有偵測的結果
            face = False
            x1 = d.left()
            y1 = d.top()
            x2 = d.right()
            y2 = d.bottom()
            if x1>0 and y1>0 and x2>0 and y2>0:
                big_size_x1 = d.left()
                big_size_y1 = d.top()
                big_size_x2 = d.right()
                big_size_y2 = d.bottom()
                face = True
            if face:
                cropped = frame[int(big_size_y1):int(big_size_y2),int(big_size_x1):int(big_size_x2)] #裁剪偵測到的人臉     
                cv2.imwrite("temporarily.jpg", cropped)
                from keras.preprocessing import image
                test_image = np.expand_dims(image.img_to_array(image.load_img("temporarily.jpg", target_size= (target_size,target_size))), 0)/255
                predict = model.predict(test_image)[0]
                predict_proba = model.predict_proba(test_image)[0]
                predict_classes = model.predict_classes(test_image)[0]
                proba = model.predict(test_image)[0][model.predict_classes(test_image)[0]]
                name = name_dict['sample'+str(model.predict_classes(test_image)[0])]
                if proba>threshold:
                    text = name+'({}%)'.format(proba)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA) #以方框標示偵測的人臉，cv2.LINE_AA為反鋸齒效果
                    cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)  #標示姓名
                else:
                    text = 'Unlabeled'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4, cv2.LINE_AA) #以方框標示偵測的人臉，cv2.LINE_AA為反鋸齒效果
                    cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)  #標示姓名
        cv2.imshow("face recognition everyone", frame)                  #顯示結果
        if cv2.waitKey(1) & 0xFF == ord('q'):                #按Q停止
            break
    cap.release()                                            #釋放資源
    cv2.destroyAllWindows()                                  #刪除任何我們建立的窗口
    
def histogram_diff(image1=None,image2=None):
    from PIL import Image
    import math
    import operator
    from functools import reduce
    h1 = Image.open(image1).histogram()
    h2 = Image.open(image2).histogram()
    diff = math.sqrt(reduce(operator.add, list(map(lambda a,b: (a-b)**2, h1, h2)))/len(h1))
    return diff
    
def face_recognition(model=None, threshold=0.9999999999999999, film=0, txt='sample_name.txt', target_size=64): 
    name_dict, number_of_samples = get_name_dict() 
    cap = cv2.VideoCapture(film)                               
    detector = dlib.get_frontal_face_detector()              
    count = 0
    while(cap.isOpened()):     
        cv2.namedWindow("face recognition", cv2.WINDOW_NORMAL)
        ret, frame = cap.read()  
        frame = cv2.flip(frame,1,dst=None)
        face_rects, scores, idx = detector.run(frame, 0)    
        big_size = 0
        big_size_idex = 0
        big_size_x1 = 0
        big_size_y1 = 0
        big_size_x2 = 0
        big_size_y2 = 0
        face = False
        for i, d in enumerate(face_rects):                  
            x1 = d.left()
            y1 = d.top()
            x2 = d.right()
            y2 = d.bottom()
            height = d.bottom()-d.top()
            width = d.right()-d.left()
            size = height*width
            if  (size > big_size) and x1>0 and y1>0 and x2>0 and y2>0:
                big_size = size
                big_size_idex = i
                big_size_x1 = d.left()
                big_size_y1 = d.top()
                big_size_x2 = d.right()
                big_size_y2 = d.bottom()
                face = True
        if face:
            cropped = frame[int(big_size_y1):int(big_size_y2),int(big_size_x1):int(big_size_x2)] #裁剪偵測到的人臉     
            cv2.imwrite("temporarily.jpg", cropped)
            from keras.preprocessing import image
            test_image = np.expand_dims(image.img_to_array(image.load_img("temporarily.jpg", target_size= (target_size,target_size))), 0)/255
            predict = model.predict(test_image)[0]
            predict_proba = model.predict_proba(test_image)[0]
            predict_classes = model.predict_classes(test_image)[0]
            proba = model.predict(test_image)[0][model.predict_classes(test_image)[0]]
            name = name_dict['sample'+str(model.predict_classes(test_image)[0])]

            if proba>threshold:
                text = name+'({}%)'.format(proba)
                cv2.rectangle(frame, (big_size_x1, big_size_y1), (big_size_x2, big_size_y2), (0, 255, 0), 4, cv2.LINE_AA) 
                cv2.putText(frame, text, (big_size_x1, big_size_y1), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)  #標示姓名
            else:
                text = 'Unlabeled'
                cv2.rectangle(frame, (big_size_x1, big_size_y1), (big_size_x2, big_size_y2), (0, 0, 255), 4, cv2.LINE_AA) #以方框標示偵測的人臉，cv2.LINE_AA為反鋸齒效果
                cv2.putText(frame, text, (big_size_x1, big_size_y1), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)  #標示姓名
        cv2.imshow("face recognition", frame)                  #顯示結果
        if cv2.waitKey(1) & 0xFF == ord('q'):                #按Q停止
            break
        count += 1
    cap.release()                                            #釋放資源
    cv2.destroyAllWindows()                                  #刪除任何我們建立的窗口

def face_recognition_system(model=None, threshold=0.9999999999999999, film=0, txt='sample_name.txt', target_size=64): 
    name_dict, number_of_samples = get_name_dict() 
    cap = cv2.VideoCapture(film)                               
    detector = dlib.get_frontal_face_detector()    
    name = None
    count = 0
    while(cap.isOpened()):     
        cv2.namedWindow("face recognition", cv2.WINDOW_NORMAL)
        ret, frame = cap.read()  
        frame = cv2.flip(frame,1,dst=None)
        face_rects, scores, idx = detector.run(frame, 0)    
        big_size = 0
        big_size_idex = 0
        big_size_x1 = 0
        big_size_y1 = 0
        big_size_x2 = 0
        big_size_y2 = 0
        face = False
        for i, d in enumerate(face_rects):                  
            x1 = d.left()
            y1 = d.top()
            x2 = d.right()
            y2 = d.bottom()
            height = d.bottom()-d.top()
            width = d.right()-d.left()
            size = height*width
            if  (size > big_size) and x1>0 and y1>0 and x2>0 and y2>0:
                big_size = size
                big_size_idex = i
                big_size_x1 = d.left()
                big_size_y1 = d.top()
                big_size_x2 = d.right()
                big_size_y2 = d.bottom()
                face = True
        if face:
            cropped = frame[int(big_size_y1):int(big_size_y2),int(big_size_x1):int(big_size_x2)] #裁剪偵測到的人臉     
            cv2.imwrite("temporarily.jpg", cropped)
            from keras.preprocessing import image
            test_image = np.expand_dims(image.img_to_array(image.load_img("temporarily.jpg", target_size= (target_size,target_size))), 0)/255
            predict = model.predict(test_image)[0]
            predict_proba = model.predict_proba(test_image)[0]
            predict_classes = model.predict_classes(test_image)[0]
            proba = model.predict(test_image)[0][model.predict_classes(test_image)[0]]
            
            Previous_name = name
            name = name_dict['sample'+str(model.predict_classes(test_image)[0])]
            if Previous_name == name:
                times += 1
            else:
                times = 0
                        
            if proba>threshold:
                text = name+'({}%)'.format(proba)
                cv2.rectangle(frame, (big_size_x1, big_size_y1), (big_size_x2, big_size_y2), (0, 255, 0), 4, cv2.LINE_AA) 
                cv2.putText(frame, text, (big_size_x1, big_size_y1), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)  #標示姓名
            else:
                text = 'Unlabeled'
                cv2.rectangle(frame, (big_size_x1, big_size_y1), (big_size_x2, big_size_y2), (0, 0, 255), 4, cv2.LINE_AA) #以方框標示偵測的人臉，cv2.LINE_AA為反鋸齒效果
                cv2.putText(frame, text, (big_size_x1, big_size_y1), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)  #標示姓名
        
        
        if times == 100:
            #紀錄時間
            
            #
            img = cv2.imread(os.path.join(os.getcwd(),'confirmation_screen','sample'+str(predict_classes)+'_face.jpg'))
            img = cv2.resize(img,frame.shape[:2][::-1],interpolation=cv2.INTER_CUBIC) #將人臉圖片大小調整為(64, 64)
            cv2.imshow("face recognition", img)     #顯示結果
            if cv2.waitKey(3000) & 0xFF == ord('q'):   #按Q停止
                break 
        
        cv2.imshow("face recognition", frame)                  #顯示結果
        if cv2.waitKey(1) & 0xFF == ord('q'):                #按Q停止
            break
        count += 1
    cap.release()                                            #釋放資源
    cv2.destroyAllWindows()                                  #刪除任何我們建立的窗口
    
def histogram_face_recognition(threshold=100, film=0, txt='sample_name.txt'): 
    name_dict, number_of_samples = get_name_dict()
    
    import os
    from PIL import Image
    import math
    import operator
    from functools import reduce
    
    sample_face = os.listdir("photograph_face")
    for i in range(len(sample_face)):
        locals()['sample%s'%i] = Image.open(os.path.join(os.getcwd(),"photograph_face",sample_face[i])).histogram()
    
    cap = cv2.VideoCapture(film)                               
    detector = dlib.get_frontal_face_detector()              
    count = 0
    while(cap.isOpened()):     
        cv2.namedWindow("face recognition", cv2.WINDOW_NORMAL)
        ret, frame = cap.read()  
        frame = cv2.flip(frame,1,dst=None)
        face_rects, scores, idx = detector.run(frame, 0)    
        big_size = 0
        big_size_idex = 0
        big_size_x1 = 0
        big_size_y1 = 0
        big_size_x2 = 0
        big_size_y2 = 0
        face = False
        for i, d in enumerate(face_rects):                  
            x1 = d.left()
            y1 = d.top()
            x2 = d.right()
            y2 = d.bottom()
            height = d.bottom()-d.top()
            width = d.right()-d.left()
            size = height*width
            if  (size > big_size) and x1>0 and y1>0 and x2>0 and y2>0:
                big_size = size
                big_size_idex = i
                big_size_x1 = d.left()
                big_size_y1 = d.top()
                big_size_x2 = d.right()
                big_size_y2 = d.bottom()
                face = True
        if face:
            cropped = frame[int(big_size_y1):int(big_size_y2),int(big_size_x1):int(big_size_x2)] #裁剪偵測到的人臉     
            cv2.imwrite("temporarily.jpg", cropped)

            tem = Image.open("temporarily.jpg").histogram()
            mim_diff = 99999
            for i in range(len(sample_face)):
                diff = math.sqrt(reduce(operator.add, list(map(lambda a,b: (a-b)**2, tem, locals()['sample%s'%i])))/len(tem))
                if diff < mim_diff:
                    mim_diff = diff  
                    name = name_dict['sample'+str(i)]
#                 print(i)
#                 print(name_dict['sample'+str(i)])
#                 print(diff) 
#                 print()
#             print("=================")
            if mim_diff<threshold:
                text = name+'(RMS:{})'.format(mim_diff)
                cv2.rectangle(frame, (big_size_x1, big_size_y1), (big_size_x2, big_size_y2), (0, 255, 0), 4, cv2.LINE_AA) 
                cv2.putText(frame, text, (big_size_x1, big_size_y1), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)  #標示姓名
            else:
                text = 'Unlabeled'
                cv2.rectangle(frame, (big_size_x1, big_size_y1), (big_size_x2, big_size_y2), (0, 0, 255), 4, cv2.LINE_AA) #以方框標示偵測的人臉，cv2.LINE_AA為反鋸齒效果
                cv2.putText(frame, text, (big_size_x1, big_size_y1), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)  #標示姓名
        cv2.imshow("face recognition", frame)                  #顯示結果
        if cv2.waitKey(1) & 0xFF == ord('q'):                #按Q停止
            break
        count += 1
    cap.release()                                            #釋放資源
    cv2.destroyAllWindows()                                  #刪除任何我們建立的窗口
    
def histogram_face_recognition_system(threshold=100, film=0, txt='sample_name.txt'): 
    name_dict, number_of_samples = get_name_dict()
    
    import os
    from PIL import Image
    import math
    import operator
    from functools import reduce
    
    sample_face = os.listdir("photograph_face")
    for i in range(len(sample_face)):
        locals()['sample%s'%i] = Image.open(os.path.join(os.getcwd(),"photograph_face",sample_face[i])).histogram()
    
    cap = cv2.VideoCapture(film)                               
    detector = dlib.get_frontal_face_detector()              
    count = 0
    times = 0
    name = None
    while(cap.isOpened()):     
        cv2.namedWindow("face recognition", cv2.WINDOW_NORMAL)
        ret, frame = cap.read()  
        frame = cv2.flip(frame,1,dst=None)
        face_rects, scores, idx = detector.run(frame, 0) 
        big_size = 0
        big_size_idex = 0
        big_size_x1 = 0
        big_size_y1 = 0
        big_size_x2 = 0
        big_size_y2 = 0
        face = False
        for i, d in enumerate(face_rects):                  
            x1 = d.left()
            y1 = d.top()
            x2 = d.right()
            y2 = d.bottom()
            height = d.bottom()-d.top()
            width = d.right()-d.left()
            size = height*width
            if  (size > big_size) and x1>0 and y1>0 and x2>0 and y2>0:
                big_size = size
                big_size_idex = i
                big_size_x1 = d.left()
                big_size_y1 = d.top()
                big_size_x2 = d.right()
                big_size_y2 = d.bottom()
                face = True
        if face:
            cropped = frame[int(big_size_y1):int(big_size_y2),int(big_size_x1):int(big_size_x2)] #裁剪偵測到的人臉     
            cv2.imwrite("temporarily.jpg", cropped)

            tem = Image.open("temporarily.jpg").histogram()
            mim_diff = 99999
            for i in range(len(sample_face)):
                diff = math.sqrt(reduce(operator.add, list(map(lambda a,b: (a-b)**2, tem, locals()['sample%s'%i])))/len(tem))
                if diff < mim_diff:
                    mim_diff = diff 
                    mim_id = i
                    Previous_name = name
                    name = name_dict['sample'+str(i)]
                    if Previous_name == name:
                        times += 1
                    else:
                        times = 0
#                 print(i)
#                 print(name_dict['sample'+str(i)])
#                 print(diff) 
#                 print()
#             print("=================")
            if mim_diff<threshold:
                text = name+'(RMS:{})'.format(mim_diff)
                cv2.rectangle(frame, (big_size_x1, big_size_y1), (big_size_x2, big_size_y2), (0, 255, 0), 4, cv2.LINE_AA) 
                cv2.putText(frame, text, (big_size_x1, big_size_y1), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)  #標示姓名
            else:
                text = 'Unlabeled'
                cv2.rectangle(frame, (big_size_x1, big_size_y1), (big_size_x2, big_size_y2), (0, 0, 255), 4, cv2.LINE_AA) #以方框標示偵測的人臉，cv2.LINE_AA為反鋸齒效果
                cv2.putText(frame, text, (big_size_x1, big_size_y1), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)  #標示姓名
        
        if times == 100:
            #紀錄時間
            
            #
            img = cv2.imread(os.path.join(os.getcwd(),'confirmation_screen','sample'+str(mim_id)+'_face.jpg'))
            img = cv2.resize(img,frame.shape[:2][::-1],interpolation=cv2.INTER_CUBIC) #將人臉圖片大小調整為(64, 64)
            cv2.imshow("face recognition", img)     #顯示結果
            if cv2.waitKey(3000) & 0xFF == ord('q'):   #按Q停止
                break
                
        cv2.imshow("face recognition", frame)                  #顯示結果
        if cv2.waitKey(1) & 0xFF == ord('q'):                #按Q停止
            break
        count += 1
    cap.release()                                            #釋放資源
    cv2.destroyAllWindows()                                  #刪除任何我們建立的窗口

