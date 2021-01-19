'''
@File    : GDELT1.py

@Modify Time        @Author         @Version    @Desciption
------------        ------------    --------    -----------
2020/12/3 下午4:22    Jichen Yeung    1.0        获取2019年原始数据，只保留20个国家的部分外交信息
'''
#coding=utf-8
import gdelt
import pandas as pd
import os

'''
这部分是为了爬取2018年GDELT的数据
'''
def get_dir_list(file_dir):
    file_list = []
    for i in os.listdir(file_dir):
        file_list.append(i)
    return(file_list)

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

# Version 1 queries
gd1 = gdelt.gdelt(version=1)
# pull events table, range, output to json format
Monlist = {'201801':['2018 1 1','2018 1 31'],'201802':['2018 2 1','2018 2 27'],'201803':['2018 3 1','2018 3 31'],
           '201804':['2018 4 1','2018 4 30'],'201805':['2018 5 1','2018 5 31'],'201806':['2018 6 1','2018 6 30'],
           '201807':['2018 7 1','2018 7 31'],'201808':['2018 8 1','2018 8 31'],'201809':['2018 9 1','2018 9 30'],
           '201810':['2018 10 1','2018 10 31'],'201811':['2018 11 1','2018 11 30'],'201812':['2018 12 1','2018 12 31']}

Infolist = ['SQLDATE', 'Actor1Geo_CountryCode', 'Actor2Geo_CountryCode', 'EventRootCode', 'Actor1Geo_Lat',
                 'Actor1Geo_Long', 'Actor2Geo_Lat', 'Actor2Geo_Long']
# result['Actor1Geo_CountryCode'].value_counts()[0:50].index
CountryList = ('US', 'IN', 'UK', 'IR', 'PK','SP', 'IS', 'CA', 'CH', 'RS', 'AS', 'FR', 'MX', 'GM', 'TU', 'RP', 'SF', 'KS','SA','BR')


if __name__=='__main__':
    i=0
    mkdir('DATA/step1_2018/')
    filelist = get_dir_list('DATA/step1_2018')
    filelist = [file[:-4] for file in filelist]
    while set(filelist) != set(Monlist.keys()):
        residual_filelist = set(Monlist.keys())-set(filelist)
        for Mon in residual_filelist:
            try:
                result = gd1.Search(Monlist[Mon], coverage=True, table='events')
                print('-------start {}-------'.format(Mon))
                result = result[Infolist]
                result = result[result['Actor1Geo_CountryCode'].isin(CountryList)]
                result = result[result['Actor2Geo_CountryCode'].isin(CountryList)]
                print('len is {}'.format(len(result)))
                print('-------end {}-------'.format(Mon))
                name = 'DATA/step1_2018/'+Mon+'.csv'
                result.to_csv(name,index = False)
            except:
                print('-------{} Wrong!-------'.format(Mon))
                continue

        filelist = get_dir_list('DATA/step1_2018')
        filelist = [file[:-4] for file in filelist]
