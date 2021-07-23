# -*- coding: utf-8 -*-
#!/usr/bin/env python
import pandas as pd
import numpy as np
from sklearn import  preprocessing
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import os
np.set_printoptions(threshold=np.NaN)
pd.set_option('display.max_rows', 2000,'display.max_columns',200)
# os.chdir("/root/user/tianchi")
# map 是作用行或者列的函数，applymap()是操作每一个元素的函数，而map()函数是作用在series的每一个元素

def haversine(lon1, lat1, lon2, lat2):
    from math import radians, cos, sin, asin, sqrt
    lon1= map(radians, np.array(lon1))
    lat1= map(radians, np.array(lat1))
    lon2= map(radians, np.array(lon2))
    lat2= map(radians, np.array(lat2))
    lon1 = np.array(list(lon1)).reshape(-1,1)
    lon2 = np.array(list(lon2)).reshape(-1,1)
    lat1 = np.array(list(lat1)).reshape(-1,1)
    lat2 = np.array(list(lat2)).reshape(-1,1)
    dlon = lon2 - lon1
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2  
    c = 2 * np.arcsin(np.sqrt(a))   
    r = 6371
    return c * r * 1000

# column1=[]
# for i in range(10):
#     column1.append("wificc{}".format(i))
# ['wificc0', 'wificc1', 'wificc2', 'wificc3', 'wificc4', 'wificc5',
# 'wificc6', 'wificc7', 'wificc8', 'wificc9']



path='D:\\data\\Datasets\\tc\\'
df=pd.read_csv(path+'ccf_first_round_user_shop_behavior.csv',engine='python')
df = df.sample(frac=0.001,random_state=22)
print len(df)
shop=pd.read_csv(path+'ccf_first_round_shop_info.csv',engine='python')
test=pd.read_csv(path+'evaluation_public.csv',engine='python')
df=pd.merge(df,shop[['shop_id','mall_id']],how='left',on='shop_id')
train=pd.concat([df,test]) # train 是联合了test的数据
train['time_stamp']=pd.to_datetime(train['time_stamp'])
# print len(train['time_stamp'])

train['isweekend']=train.time_stamp.map(lambda x:1 if x.weekday()>4 else 0)      # 判断是否是周末
dfx1=df.set_index("shop_id").wifi_infos.str.split(";",expand=True)               # shop_id 放在最前面
dfx1=dfx1.fillna("l").applymap(lambda x:x[:x.find("|")] if "b" in x else np.nan) # 把每一个shop_id分出来了
dfx2=dfx1[0] # 取第0列


# 每个商店对应的所有的wifi_id 弄成列
for i in dfx1.columns[1:]: # 其实是从1-9开始计数
    dfx3=dfx1[i]
    dfx2=pd.concat([dfx2,dfx3]) # 所有的行拼接成列
dfx2.dropna(inplace=True)# 去掉空值

# print dfx2.value_counts()[dfx2.value_counts()>1] # 这段代码来表达符合某个条件的值显示出来！
# print dfx2.value_counts()[dfx2.value_counts()>1].index # 这段代码来表达符合某个条件的值显示出来！

# dfx2 是把所有的商店出现的wifi次数大于1的表示出来。
# print 'after dfx2 :\n',dfx2.groupby(level=0).apply(lambda x:np.array(x.value_counts()[x.value_counts()>1].index))

# 除了shop之外，还构成了一个新的特征，wifi1是出现次数大于1的WiFi名称。
shop=pd.merge(shop,pd.DataFrame(dfx2.groupby(level=0).apply(lambda x:np.array(x.value_counts()[x.
                value_counts()>1].index))).rename(columns={0:"wifi1"}),left_on="shop_id",right_index=True)

# 除了shop之外，还构成了一个新的特征，wifi2是出现次数大于10的WiFi名称。
shop=pd.merge(shop,pd.DataFrame(dfx2.groupby(level=0).apply(lambda x:np.array(x.value_counts()[x.
                value_counts()>10].index))).rename(columns={0:"wifi2"}),left_on="shop_id",right_index=True)

# 除了shop之外，还构成了一个新的特征，wifi2是出现次数大于20的WiFi名称。
shop=pd.merge(shop,pd.DataFrame(dfx2.groupby(level=0).apply(lambda x:np.array(x.value_counts()[x.
                value_counts()>20].index))).rename(columns={0:"wifi3"}),left_on="shop_id",right_index=True)

# 新特征，wific是shop_id 的所有的wifif的名称。
shop=pd.merge(shop,pd.DataFrame(dfx2.groupby(level=0).apply(lambda x:np.array(x.value_counts().index)))
              .rename(columns={0:"wific"}),left_on="shop_id",right_index=True)


lbl = preprocessing.LabelEncoder()
shop["category_id"]=lbl.fit_transform(shop.category_id.values)
result=pd.DataFrame()

df['time_stamp']=pd.to_datetime(df['time_stamp'])
df["hour"]=df.time_stamp.map(lambda x:x.hour) # df 特征再加一个时间hour

dfz1=df.groupby(["shop_id","hour"]).shop_id.count().unstack().fillna(0)

# print 'the dfz1 is:\n',dfz1 #只有shop_id 是保留的，column是时间上从1到24之间取它有的数,并不连续！
# dfz2=df.groupby(["mall_id","hour"]).shop_id.count().unstack().fillna(0)
dfz3 = df.groupby(["mall_id","hour"]).shop_id.count().unstack()

# shopx=pd.merge(shop.set_index("shop_id"),df.groupby(["mall_id","hour"]).shop_id.count().unstack(),
#                left_on="mall_id",right_index=True)

shopx=pd.merge(shop.set_index("shop_id"),dfz3,left_on="mall_id",right_index=True) # 按照mall_id合并

# print shopx # [938 rows x 27 columns]
# print 'the dfz1 index is:\n',dfz1.index # dfz1.index 是长度为936的所有的shop_id的名字的字典
#[u's_1005343', u's_10076', u's_101080', u's_1017', u's_10288',u's_1035004', u's_103705',····]
# print shopx.loc[:,range(24)].loc[dfz1.index] # 这里是按照所给的dfz1的 shop_id 序列来选择行的顺序
dfzz=(dfz1/shopx.loc[:,range(24)].loc[dfz1.index]).fillna(0) # 从0到23



for i in shop.mall_id.unique(): # 商场中的 shop_id 信息的唯一值。

    train1=train[train.mall_id==i]
    train1=train1.reset_index().drop("index",axis=1) # index 重新排列
    shop1=shop[shop.mall_id==i] # 所有相同的mall_id
    shop1.columns=shop1.columns.map(lambda x:x+"1") # 列加上1
    # xx=train1.wifi_infos.str.split(";",expand=True).fillna("l").applymap(lambda x:x.split("|")[0] if "b" in x else np.nan)
    t1 = train1.wifi_infos.str.split(";",expand=True) # 相同的mall_id里面的wifi信息
    t2 = t1.fillna('l')
    xx = t2.applymap(lambda x:x.split("|")[0] if "b" in x else np.nan)
    dfy2 = xx[0]
    t3 = train1.wifi_infos.str.split(";",expand=True).fillna("l")
    t4 = t3.applymap(lambda x:x.split("|")[1] if "b" in x else np.nan) # 强度的值
    yy=t4.astype(float) # wifi 的强度
#
    for i in xx.columns[1:]:
        dfy3=xx[i]
        dfy2=pd.concat([dfy2,dfy3]) # 这一步是把所有xx的列都叠在一起
    # print dfy2.value_counts()[dfy2.value_counts()>20].index
    indexx=dfy2.value_counts()[dfy2.value_counts()>20].index # 所有wifi出现个数大于20的wifi名找出来
    xx=xx.applymap(lambda x:x if x in indexx else np.nan) # 大于20的wifi数目，否则为nan
    # xx这里是所有大于连接次数大于20的wifi的名称
    yy[xx.isnull()]=np.nan
    # print 'before xx is:\n',xx
    t5 = xx.apply(lambda x: dict(zip(x.dropna().values.tolist(),yy.loc[x.name].dropna().values.tolist())),axis=1)
    # print t5 # wifi名+强度
    t6 = pd.DataFrame(t5)
    # train1 又加上了1列特征，t6
    train1=pd.merge(train1,t6.rename(columns={0:"wifid"}),left_index=True,right_index=True)
    dfy2=dfy2.map(lambda x:x if x in indexx else np.nan)
    dfy2.dropna(inplace=True)

    t7 = pd.DataFrame(dfy2.groupby(level=0).apply(lambda x:x.values)).rename(columns={0:"wifix"})
    train1=pd.merge(train1,t7,left_index=True,right_index=True)
    train2=pd.merge(train1,shop1,left_on="mall_id",right_on="mall_id1")
    # print train2
    # 加上了一列特征‘away’
    train2["away"]=haversine(train2.longitude1,train2.latitude1,train2.longitude,train2.latitude)
    train2["wificount1"]=train2.wifi11.map(lambda x:len(x))
    train2["wificount2"]=train2.wifi21.map(lambda x:len(x))
    train2["wificount3"]=train2.wifi31.map(lambda x:len(x))
    train2["wificount4"]=train2.wific1.map(lambda x:len(x))
    train2[["count1","count2","count3"]]=train2.apply(lambda x:x[["wifi11","wifi21","wifi31"]].
            map(lambda y:len(np.intersect1d(y,x.wifix))/len(x.wifix)),axis=1)

    train2["hour"]=train2.time_stamp.map(lambda x:x.hour)
    train2["hourco"]=train2.apply(lambda x:dfzz.loc[x.shop_id1,x.hour],axis=1)
    # print 'the train2 is:\n',train2


    def fun(x):
        dic=x.wifid
        ss=pd.Series(np.intersect1d(x.wific1,x.wifix))
        ss=ss.map(dic).sort_values(ascending=False).values.tolist()
        if len(ss)<10:
            ss.extend([""]*(10-len(ss)))
        return "{},{},{},{},{},{},{},{},{},{}".format(ss[0],ss[1],ss[2],ss[3],ss[4],ss[5],ss[6],
                                                      ss[7],ss[8],ss[9])




    train2["wifico"]=train2.apply(fun,axis=1)
    # print train2.columns
    trainx=train2.wifico.str.split(",",expand=True).applymap(lambda x:np.nan if len(x)==0 else x).astype(float)

    column1 = []
    for i in range(10):
        column1.append("wificc{}".format(i))
    # ['wificc0', 'wificc1', 'wificc2', ··· 'wificc9']
    trainx.columns=column1
    trainx["wifimean"]=trainx.mean(1)
    trainx["wifisum"]=trainx.sum(1)
    trainx["wificount"]=trainx.count(1)
    train2=pd.merge(train2,trainx,left_index=True,right_index=True)
    feature=['wificount','category_id1','price1','away','wificount1','wificount2','wificount4','count1',
             'count2','count3','hourco','wifimean','wifisum','isweekend']
    feature.extend(column1)
    train3=train2[train2.shop_id.notnull()]
    test=train2[train2.shop_id.isnull()]
    train3["label"]=(train3.shop_id==train3.shop_id1)*1
    train4=train3.copy()
    print 'the train4 info is:\n',train4.info()
    print 'the train4 label is:\n',train4.label
    # model=xgb.XGBClassifier(max_depth=6,learning_rate=0.1,n_estimators=100)
    # params=model.get_xgb_params()
    # xgbtrain=xgb.DMatrix(train4[feature],train4.label, missing = -999.0)
    # cvresult=xgb.cv(params,xgbtrain,num_boost_round=1000,nfold=5,metrics="auc",early_stopping_rounds=50)
    # print(cvresult)
    # model.set_params(n_estimators=cvresult.shape[0])
    # model.fit(train4[feature],train4.label)
    # test["label"]=model.predict_proba(test[feature])[:,1]
    # print 'the label is:\n',test["label"]
    # model=lgb.LGBMClassifier(boosting_type='gbdt',
    #                          num_leaves=80,
    #                          max_depth=-1,
    #                          learning_rate=0.1,
    #                          n_estimators=10,
    #                          max_bin=255,
    #                          subsample_for_bin=50000,
    #                          objective="binary",
    #                          min_split_gain=0.0,
    #                          min_child_weight=5,
    #                          min_child_samples=10,
    #                          subsample=1.0,
    #                          subsample_freq=1,
    #                          colsample_bytree=1.0,
    #                          reg_alpha=0.0, reg_lambda=0.0,
    #                          random_state=0, n_jobs=-1, silent=True)
    #
    # params=model.get_params()
    # params.pop("n_estimators")
    # params.pop("silent")
    # params.pop("n_jobs")
    # params.pop("random_state")
    # lgbtrain=lgb.Dataset(train4[feature].values,train4.label.values.tolist())
    # res=lgb.cv(params,lgbtrain,metrics="auc",num_boost_round=1000,nfold=5,early_stopping_rounds=50)
    # print(pd.DataFrame(res))
    # model.set_params(n_estimators=pd.DataFrame(res).shape[0])
    # model.fit(train4[feature],train4.label)
    # test["label1"]=model.predict_proba(test[feature])[:,1]
    # r=test[['row_id','shop_id1',"label","label1"]]
    # r.columns=["row_id","shop_id","label","label1"]
    # result=pd.concat([result,r])
    # result['row_id']=result['row_id'].astype('int')
    # result.to_csv(path+'suba0.csv',index=False)









