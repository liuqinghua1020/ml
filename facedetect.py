#coding=utf-8
import urllib, urllib2, sys
import ssl
import base64
import json
import numpy as np
from numpy import linalg as la

#欧式距离
def euclidSimilar(inA,inB):
    return 1.0/(1.0+la.norm(inA - inB))
#皮尔逊相关系数
def pearsonSimilar(inA,inB):
    if len(inA)<3:
        return 1.0
    return 0.5+0.5*np.corrcoef(inA,inB,rowvar=0)[0][1]
#余弦相似度
def cosSimilar(inA,inB):
    inA=np.mat(inA)
    inB=np.mat(inB)
    num=float(inA*inB.T)
    denom=la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom)

host = 'https://dm-24.data.aliyun.com'
path = '/rest/160601/face/feature_detection.json'
method = 'POST'
appcode = ''
querys = ''
url = host + path
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

#读取第一幅图片
input1 = open('kuli.jpeg', 'rb')
buffer1 = input1.read();
str1 = base64.b64encode(buffer1)

input2 = open('kuli2.jpg', 'rb')
buffer2 = input2.read();
str2 = base64.b64encode(buffer2)

input3 = open('zhanmusi.jpg', 'rb')
buffer3 = input3.read();
str3 = base64.b64encode(buffer3)

bodys = {}
bodysJson = {
    "inputs":[
                  {
                    "image":{"dataType":50,"dataValue":str1},
                    "type":{"dataType":10,"dataValue":4}
                  },
                  {
                    "image":{"dataType":50,"dataValue":str3},
                    "type":{"dataType":10,"dataValue":4}
                  }
             ]
}

bodys[''] = json.dumps(bodysJson)
print (bodys['']);
#bodys1[''] = "{\"inputs\":[{\"image\":{\"dataType\":50,\"dataValue\":\"" + str1 + "\"}}]}"
post_data1 = bodys['']
request = urllib2.Request(url, post_data1)
request.add_header('Authorization', 'APPCODE ' + appcode)
#根据API的要求，定义相对应的Content-Type
request.add_header('Content-Type', 'application/json; charset=UTF-8')
response = urllib2.urlopen(request, context=ctx)
content = response.read()
if (content):
    print(content)
    result = json.loads(content)

#获取dense的数组
dense1 = json.loads(result['outputs'][0]['outputValue']['dataValue'])
dense2 = json.loads(result['outputs'][1]['outputValue']['dataValue'])

print dense1['dense']
print dense2['dense']

#将脸部特征向量转为numpy数组
arrdense1 = dense1['dense']
arrdense2 = dense2['dense']

#欧式距离
#print(euclidSimilar(arrdense1, arrdense2))

#皮尔逊相关系数
print(pearsonSimilar(arrdense1, arrdense2))

#余弦相似度
print(cosSimilar(arrdense1, arrdense2))
