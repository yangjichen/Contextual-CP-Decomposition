'''
@File    : GDELT2.py

@Modify Time        @Author         @Version    @Desciption
------------        ------------    --------    -----------
2020/12/3 下午5:31    Jichen Yeung    1.0        将原始数据整理为张量保存，且将每个地区经纬度保存
'''

import pandas as pd
import numpy as np
import os
import datetime

def mkdir(path):
    """
    :param path: 创建文件夹
    :return: 创建文件夹
    """
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder are created  ---")

    else:
        print("---  This folder already exists  ---")

def get_dir_list(file_dir):
    file_list = []
    for i in os.listdir(file_dir):
        file_list.append(i)
    return(file_list)
def build_tensor(ori_data,printitn=10000):
    data = ori_data[['SQLDATE', 'Actor1Geo_CountryCode', 'Actor2Geo_CountryCode']]
    # -----step1 清理数据-----
    # 删除时间不合适的值/将DATA转化为月份
    data = data[(data['SQLDATE'] < para['end']) & (data['SQLDATE'] > para['start'])]
    data['SQLDATE'] = data['SQLDATE'].astype(str)
    data['SQLDATE'] = data['SQLDATE'].apply(lambda x: x[4:6]).astype(int)

    # 删除地理信息同时为na的行
    a = pd.isna(data['Actor1Geo_CountryCode']) & pd.isna(data['Actor2Geo_CountryCode'])
    data = data[~a]

    # ------step2 构建列表，且删除nan值，这里为了避免麻烦的排序，手动按照step1中的排序输入geolist顺序-------
    geo_list = ['US', 'IN', 'UK', 'IR', 'PK', 'SP', 'IS', 'CA', 'CH', 'RS', 'AS', 'FR', 'MX', 'GM', 'TU', 'RP', 'SF', 'KS','SA','BR']
    time_list = list(pd.unique(data['SQLDATE']))
    time_list = np.sort([x for x in time_list if str(x) != 'nan'])

    dict1 = dict(zip(geo_list, range(len(geo_list))))
    dict2 = dict(zip(np.sort(time_list), range(len(time_list))))
    #-------step3 构建张量方法一，Groupby-------
    Tensordata = np.zeros( (len(geo_list), len(geo_list), len(time_list)))
    grouped = data.groupby(['Actor1Geo_CountryCode', 'Actor2Geo_CountryCode', 'SQLDATE']).size().reset_index(name='counts')
    for i in range(len(grouped)):
        if (not pd.isna(grouped.iloc[i]['Actor1Geo_CountryCode'])) and (not pd.isna(grouped.iloc[i]['Actor2Geo_CountryCode'])):
            location1 = grouped.iloc[i]['Actor1Geo_CountryCode']
            location2 = grouped.iloc[i]['Actor2Geo_CountryCode']
        else:
            continue
        time = grouped.iloc[i]['SQLDATE']
        Tensordata[dict1[location1], dict1[location2], dict2[time]] = grouped.iloc[i]['counts']

    """
    #-------step3 构建张量方法二：所用时间长-------
    tensorr = np.zeros( (len(geo_list), len(geo_list), len(time_list)))
    for i in range(len(data)):
        if pd.isna(data.iloc[i]['Actor1Geo_CountryCode']) and pd.isna(data.iloc[i]['Actor2Geo_CountryCode']):
            location1 = data.iloc[i]['Actor1Geo_CountryCode']
            location2 = data.iloc[i]['Actor1Geo_CountryCode']
        else:
            continue
        time = data.iloc[i]['SQLDATE']

        if (i + 1) % printitn == 0:
            print ('Finish: iterations={0}, Percentage={1}'.format(i+1, (i+1)/len(data)))

        tensorr[dict1[location1], dict1[location2], dict2[time]] += 1
    """
    return(Tensordata,geo_list,time_list)

if __name__ == '__main__':
    para = {'start':20180101, 'end':20181231}
    # -------读取数据start--------
    filelist = get_dir_list('DATA/step1_2018')
    filelist = ['DATA/step1_2018/'+filelist[i] for i in range(len(filelist))]
    print(filelist)
    dataset = pd.DataFrame()
    for i in filelist:
        dataset = dataset.append(pd.read_csv(i))
        print('-----read data{}-----'.format(i))
    print('Lenth of data is {}'.format(len(dataset)))
    # -------读取数据end--------


    # -------构造张量start--------
    Tensordata,geo_list,time_list = build_tensor(dataset)
    print(geo_list)
    print(time_list)
    mkdir('DATA/step2/')
    np.save('DATA/step2/tensor.npy', Tensordata)
    np.save('DATA/step2/geolist.npy', geo_list)
    np.save('DATA/step2/timelist.npy', time_list)
    # -------构造张量end--------


    # -------获取经纬度start--------
    Lat_Long = []
    for geo in geo_list:
        tmp = dataset[dataset['Actor1Geo_CountryCode']==geo]
        Lat_Long.append(tmp.iloc[0][['Actor1Geo_Lat', 'Actor1Geo_Long']].tolist())
    np.save('DATA/step2/Lat_Long.npy', Lat_Long)
    # -------获取经纬度end--------


