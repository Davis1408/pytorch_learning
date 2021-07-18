# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 16:16:39 2017

@author: 麦芽的香气
"""
import warnings
import pandas as pd
import numpy as np
from sklearn import  preprocessing
import xgboost as xgb
import lightgbm as lgb
warnings.filterwarnings(action='ignore')
# 数据准备
path='C:\\Users\\bruce\\Desktop\\'
df=pd.read_csv(path+u'训练数据-ccf_first_round_user_shop_behavior.csv')
shop=pd.read_csv(path+u'训练数据-ccf_first_round_shop_info.csv') # info 数据
test=pd.read_csv(path+'evaluation_public.csv')

# 数据处理
df=pd.merge(df,shop[['shop_id','mall_id']],how='left',on='shop_id') # 加入mall_id

# df.to_csv('C:\\Users\\bruce\\Desktop\\trainmerge.csv')
# 2017-08-29 13:40 时间改变之后 2017-08-29 13:40:00
df['time_stamp']=pd.to_datetime(df['time_stamp'])
train=pd.concat([df,test])
# train.to_csv('C:\\Users\\bruce\\Desktop\\traintest.csv')
mall_list=list(set(list(shop.mall_id))) # 97 个商场
result=pd.DataFrame()

#
for mall in mall_list:
    train1=train[train.mall_id==mall].reset_index(drop=True) # index 从0开始递增加1计数
    # train1.to_csv('C:\\Users\\bruce\\Desktop\\train1mall.csv')
    l=[]
    wifi_dict = {}
    for index,row in train1.iterrows(): # 行号和内容  迭代每一个商场的所有样本
        r = {}
        wifi_list = [wifi.split('|') for wifi in row['wifi_infos'].split(';')]
        # print wifi_list
        for i in wifi_list:
            r[i[0]]=int(i[1]) # r是一个字典，wifi名是键，值是强度
            if i[0] not in wifi_dict:
                wifi_dict[i[0]]=1
            else:
                wifi_dict[i[0]]+=1
        l.append(r) # l是 一个mall 每一个店铺wifi和强度的字典

    # wifi 删除操作
    delate_wifi=[]
    # print wifi_dict
    for i in wifi_dict: # wifi_dict 记录的书每个mall里面的所有客户连接wifi出现次数
        # print i
        if wifi_dict[i]<20:
            delate_wifi.append(i) # 连接个数小于20的wifi名字
    m=[]
    for row in l:
      #  print row # {'b_15322575': -86, 'b_49978743': -74, 'b_38348513': -86, 'b_36487312': -73, 'b_38407258': -43, 'b_38407259': -40}
        new={}
        for n in row.keys():
            if n not in delate_wifi:
                new[n]=row[n]
        m.append(new) # wifi 连接数大于20的wifi名
    # print m # 得到一个列表，元素都是字典，wifi信息出现个数大于20的字典，值是它的强度。···b_39379387': -68}, {'b_39382912···

    train1 = pd.concat([train1,pd.DataFrame(m)], axis=1)
    # train1.to_csv('C:\\Users\\bruce\\Desktop\\train2mall.csv')

    df_train=train1[train1.shop_id.notnull()] # 把不是空的shop_id选出来作为训练集
    df_test=train1[train1.shop_id.isnull()]   # 这是空的shop_id 选出来作为验证集
    lbl = preprocessing.LabelEncoder()
    # print df_train['shop_id']
    # print df_train['shop_id'].values
    # print list(df_train['shop_id'].values)
    lbl.fit(list(df_train['shop_id'].values))
    df_train['label'] = lbl.transform(list(df_train['shop_id'].values))
    # print df_train['label']
    # print df_train
    # df_train.to_csv('C:\\Users\\bruce\\Desktop\\train_label.csv')
    num_class=df_train['label'].max()+1 # 统计一共是几分类

    params = {
            'objective': 'multi:softmax',
            'eta': 0.05, #0.1
            'max_depth': 6,
            'eval_metric': 'merror',
            'seed': 0,
            'missing': -999,
            'num_class':num_class,
            'silent' : 1
            }
    feature=[x for x in train1.columns if x not in ['user_id','label','shop_id','time_stamp',
                                                    'mall_id','wifi_infos']]
    print feature
    # 训练模型 + 参数设置
    xgbtrain = xgb.DMatrix(df_train[feature], df_train['label'], missing = np.nan)# 这里加上missing = np.nan
    xgbtest = xgb.DMatrix(df_test[feature],missing = np.nan)
    watchlist = [ (xgbtrain,'train'), (xgbtrain, 'test') ]
    num_rounds=200
    model = xgb.train(params, xgbtrain, num_rounds, watchlist, early_stopping_rounds=60)
    #
    df_test['label']=model.predict(xgbtest)
    df_test['shop_id']=df_test['label'].apply(lambda x:lbl.inverse_transform(int(x)))
    r=df_test[['row_id','shop_id']]
    result=pd.concat([result,r])
    result['row_id']=result['row_id'].astype('int')
    result.to_csv(path+'sub.csv',index=False)