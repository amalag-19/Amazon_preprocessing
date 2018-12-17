import numpy as np
import pandas as pd
import gzip
import gc
import re

##### READ IN ROW DATAFRAME FROM JSON FILE

def parse(path):
    g = open(path, 'rb')
    for l in g: yield eval(l)


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

    
"""temp={"a":1.,"b":2.,"c":3.}
pd.DataFrame.from_dict(temp,orient='index')
k=list(temp.keys())
k    
k.sort()
k
for i in k: print i, temp[i]"""


    
#df = getDF('E:/R/input/reviews_Clothing_Shoes_and_Jewelry_5.json')

path1='/Users/Amal/Box Sync/PSU/Fall 2018/Main_Research/Network Models/Project 5 (Bipartite)/Real Data/bipartite network/reviews_Clothing_Shoes_and_Jewelry_5.json'
df = getDF(path1)

df.to_csv("/Users/Amal/Box Sync/PSU/Summer 2018/Main_Research/Network Models/Project 5 (Bipartite)/amazon_5core/edge_list.csv",index = False)


edge_list = df.loc[:,['reviewerID','asin','reviewText','summary','reviewTime','overall']]
edge_list['reviewTime'] = edge_list['reviewTime'].apply(lambda x: re.split(r'[,\s]+',x))

month = pd.Series(range(len(edge_list)))
year = pd.Series(range(len(edge_list)))

i = 0 
for l in edge_list['reviewTime']:
    month[i] = l[1]
    year[i] = l[2]
    i = i + 1

edge_list["month"] = month
edge_list["year"] = year
del edge_list['reviewTime']
edge_list.to_csv("/Users/Amal/Box Sync/PSU/Summer 2018/Main_Research/Network Models/Project 5 (Bipartite)/amazon_5core/edge_list.csv",index = False)

reviewer = pd.DataFrame({'reviewerID':pd.Series(df.groupby('reviewerID')['overall'].mean().index)})
reviewer['num_product']=pd.Series(df.groupby('reviewerID')['asin'].count().values)
reviewer['mean_rating']=pd.Series(df.groupby('reviewerID')['overall'].mean().values)
reviewer.to_csv("/Users/Amal/Box Sync/PSU/Summer 2018/Main_Research/Network Models/Project 5 (Bipartite)/amazon_5core/reviewer_list.csv",index = False)

product = pd.DataFrame({'asin':pd.Series(df.groupby('asin')['overall'].mean().index)})
product['num_reviewer']=pd.Series(df.groupby('asin')['reviewerID'].count().values)
product['mean_rating']=pd.Series(df.groupby('asin')['overall'].mean().values)
product.to_csv("/Users/Amal/Box Sync/PSU/Summer 2018/Main_Research/Network Models/Project 5 (Bipartite)/amazon_5core/product_list.csv",index = False)

df.info()

## For 2014

np.max(year)
#pd.DataFrame(edge_list[:,""])

edge_list_2014=edge_list.loc[edge_list['year'] == np.max(year),['reviewerID', 'asin','overall']]
edge_list_2014.to_csv("/Users/Amal/Box Sync/PSU/Spring 2018/Main_Research/Network Models/Project 5 (Bipartite)/bipartite network/edge_list_bipartite_2014.csv",index = False)

## Reviewer list for 2014
reviewer_2014 = pd.DataFrame({'reviewerID':pd.Series(edge_list_2014.groupby('reviewerID')['overall'].mean().index)})
reviewer_2014['num_product']=pd.Series(edge_list_2014.groupby('reviewerID')['asin'].count().values)
reviewer_2014['mean_rating']=pd.Series(edge_list_2014.groupby('reviewerID')['overall'].mean().values)
reviewer_2014['median_rating']=pd.Series(edge_list_2014.groupby('reviewerID')['overall'].median().values)
reviewer_2014.to_csv("/Users/Amal/Box Sync/PSU/Summer 2018/Main_Research/Network Models/Project 5 (Bipartite)/Real Data/amazon_5core/2014/reviewer_cov_2014.csv",index = False)

## Product list for 2014
product_2014 = pd.DataFrame({'asin':pd.Series(edge_list_2014.groupby('asin')['overall'].mean().index)})
product_2014['num_reviewer']=pd.Series(edge_list_2014.groupby('asin')['reviewerID'].count().values)
product_2014['mean_rating']=pd.Series(edge_list_2014.groupby('asin')['overall'].mean().values)
product_2014['median_rating']=pd.Series(edge_list_2014.groupby('asin')['overall'].median().values)
product_2014.to_csv("/Users/Amal/Box Sync/PSU/Summer 2018/Main_Research/Network Models/Project 5 (Bipartite)/Real Data/amazon_5core/2014/product_cov_2014.csv",index = False)

reviewer_2014.info()
product_2014.info()

reviews_year_summary=pd.DataFrame({'year':pd.Series(edge_list.groupby('year')['year'].count().index)})
reviews_year_summary['count']=pd.Series(edge_list.groupby('year')['year'].count().values)

reviews_year_summary

################################################################################################
path1='/Users/Amal/Box Sync/PSU/Summer 2018/Main_Research/Network Models/Project 5 (Bipartite)/Real Data/bipartite network/meta_Clothing_Shoes_and_Jewelry.json'
df = getDF(path1)


all_asins = df['asin']
all_asins.to_csv("/Users/Amal/Box Sync/PSU/Summer 2018/Main_Research/Network Models/Project 5 (Bipartite)/bipartite network/all_asins.csv",index = False)




women_asins = df[df["categories"].apply(str).str.contains("Women")]['asin']


df['categories'][100]
type(df['categories'][1])

df["categories"][0]

elements = [10, 11, 12, 13, 14, 15]
indices = (1,2,4,5)

result_list = [df["categories"][k] for k in indices]  

result_list[2]
df["categories"][4]


import csv
with open("/Users/Amal/Box Sync/PSU/Summer 2018/Main_Research/Network Models/Project 5 (Bipartite)/bipartite network/asins_row_IDs.csv") as f:
    reader = csv.reader(f)
    next(reader) # skip header
    row_IDs = [r for r in reader]

len(row_IDs)

row_IDs[0][1]

row_IDs_list = [int(row_IDs[k][1]) for k in range(0,1749)] 

#type(row_IDs_list[0])

#int(row_IDs_list[0])

row_IDs_list[0:10]

result_categories = [df["categories"][k] for k in row_IDs_list]  


with open("/Users/Amal/Box Sync/PSU/Summer 2018/Main_Research/Network Models/Project 5 (Bipartite)/bipartite network/result_categories.csv",'wb') as resultFile:
    wr = csv.writer(resultFile, dialect='excel')
    wr.writerows(result_categories)
    
################################################################################################
## Getting the price and salesRank
import csv
df_cov = df[["asin","price","salesRank","categories"]]
df_cov.to_csv("/Users/Amal/Box Sync/PSU/Summer 2018/Main_Research/Network Models/Project 5 (Bipartite)/Real Data/amazon_5core/df_price_salesRank.csv")

################################################################################################
import time as time

temp=time.ctime(df['unixReviewTime'][1])

time.strftime("%d %m, %Y",temp)

df_whole = pd.read_csv('/Users/Amal/Box Sync/PSU/Summer 2018/Main_Research/Network Models/Project 5 (Bipartite)/data/ratings_Clothing_Shoes_and_Jewelry.csv', dtype='object',header=0)

df_whole.info()


df_whole['date']=pd.to_datetime(df_whole['reviewTime'],unit='s')

df_whole['year'] = df_whole['date'].dt.year

df_whole.head()[1:5]

np.max(df_whole['year'])
#pd.DataFrame(edge_list[:,""])

edge_list_2014=df_whole.loc[df_whole['year'] == np.max(df_whole['year']),['reviewerID', 'asin','rating']]
edge_list_2014.to_csv("/Users/Amal/Box Sync/PSU/Summer 2018/Main_Research/Network Models/Project 5 (Bipartite)/amazon_whole_2014/edge_list_bipartite_2014.csv",index = False)

edge_list_2014['rating']=pd.to_numeric(edge_list_2014['rating'])


## Reviewer list for 2014
reviewer_2014 = pd.DataFrame({'reviewerID':pd.Series(edge_list_2014.groupby('reviewerID')['rating'].mean().index)})
reviewer_2014['num_product']=pd.Series(edge_list_2014.groupby('reviewerID')['asin'].count().values)
reviewer_2014['mean_rating']=pd.Series(edge_list_2014.groupby('reviewerID')['rating'].mean().values)
reviewer_2014['median_rating']=pd.Series(edge_list_2014.groupby('reviewerID')['rating'].median().values)
reviewer_2014.to_csv("/Users/Amal/Box Sync/PSU/Summer 2018/Main_Research/Network Models/Project 5 (Bipartite)/amazon_whole_2014/reviewer_list_2014.csv",index = False)

reviewer_2014.info()

## Product list for 2014
product_2014 = pd.DataFrame({'asin':pd.Series(edge_list_2014.groupby('asin')['rating'].mean().index)})
product_2014['num_reviewer']=pd.Series(edge_list_2014.groupby('asin')['reviewerID'].count().values)
product_2014['mean_rating']=pd.Series(edge_list_2014.groupby('asin')['rating'].mean().values)
product_2014['median_rating']=pd.Series(edge_list_2014.groupby('asin')['rating'].median().values)
product_2014.to_csv("/Users/Amal/Box Sync/PSU/Summer 2018/Main_Research/Network Models/Project 5 (Bipartite)/amazon_whole_2014/product_list_2014.csv",index = False)

product_2014.info()





del edge_list['reviewTime']
edge_list.to_csv("../edge_list_bipartite.csv",index = False)




