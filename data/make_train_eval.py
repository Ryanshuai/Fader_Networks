import os
import numpy as np
import shutil
import cv2

img_path = 'img_align_celeba/'
attr_file = open('list_attr_celeba.txt')
attr_all_num = int(attr_file.readline())
attr_names = attr_file.readline()
names = [na for na in attr_names.split()]
print(len(names))


# train
# make dir
train_path = 'train'
train_img_path = os.path.join(train_path, 'img')
if os.path.exists(train_path) == False:
    os.mkdir(train_path)
if os.path.exists(train_img_path) == False:
    os.mkdir(train_img_path)
# move data
train_num = 162770
train_attr_file = open(train_path + '/attributes.txt','w')
train_attr_file.truncate()#清空文件内容
train_attr_file.write(str(train_num)+"\n")
train_attr_file.write(attr_names)
for i in range(train_num):
    i = i + 1
    train_attr_file.write(attr_file.readline())
    img = cv2.imread(img_path+str(i).zfill(6)+".jpg")
    img = cv2.resize(img, (256,256))
    cv2.imwrite(train_img_path+'/'+str(i).zfill(6)+'.jpg', img)
train_attr_file.close()
#check data
# train_attr_file = open(train_path + '/attributes.txt','r')
# for i, name in enumerate(train_attr_file.readlines()):
#     print(i,'--',name)
# for i, name in enumerate(os.listdir(train_img_path)):
#     print(i, '--', name)

# val
# make dir
val_path = 'val'
val_img_path = os.path.join(val_path, 'img')
if os.path.exists(val_path) == False:
    os.mkdir(val_path)
if os.path.exists(val_img_path) == False:
    os.mkdir(val_img_path)
# move data
val_num = 19867
val_attr_file = open(val_path + '/attributes.txt','w')
val_attr_file.truncate()#清空文件内容
val_attr_file.write(str(val_num)+"\n")
val_attr_file.write(attr_names)
for i in range(val_num):
    i = i + 162770 + 1
    val_attr_file.write(attr_file.readline())
    img = cv2.imread(img_path+str(i).zfill(6)+".jpg")
    img = cv2.resize(img, (256,256))
    cv2.imwrite(val_img_path+'/'+str(i).zfill(6)+'.jpg', img)# str.zfill(6): "1"->"000001"
val_attr_file.close()
#check data
# val_attr_file = open(val_path + '/attributes.txt','r')
# for i, name in enumerate(val_attr_file.readlines()):
#     print(i,'--',name)
# for i, name in enumerate(os.listdir(val_img_path)):
#     print(i, '--', name)

# test
# make dir
test_path = 'test'
test_img_path = os.path.join(test_path, 'img')
if os.path.exists(test_path) == False:
    os.mkdir(test_path)
if os.path.exists(test_img_path) == False:
    os.mkdir(test_img_path)
# move data
test_num = attr_all_num - 162770 - 19867
test_attr_file = open(test_path + '/attributes.txt','w')
test_attr_file.truncate()#清空文件内容
test_attr_file.write(str(test_num)+"\n")
test_attr_file.write(attr_names)
for i in range(test_num):
    i += 162770 + 19867 + 1 # 19962
    test_attr_file.write(attr_file.readline())
    img = cv2.imread(img_path+str(i).zfill(6)+".jpg")
    img = cv2.resize(img, (256, 256))
    cv2.imwrite(test_img_path+'/'+str(i).zfill(6)+'.jpg', img)  # str.zfill(6): "1"->"000001"
test_attr_file.close()
#check data
# test_attr_file = open(test_path + '/attributes.txt','r')
# for i, name in enumerate(test_attr_file.readlines()):
#     print(i,'--',name)
# for i, name in enumerate(os.listdir(test_img_path)):
#     print(i, '--', name)