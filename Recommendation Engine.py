#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')


# In[3]:


movies.head()


# In[4]:


ratings.head()


# In[5]:


ratings = ratings.drop('timestamp',1)


# In[6]:


ratings.head()


# # Creating the user profile

# This function is for making the user profile to which the movies will be recommended to

# In[7]:


user = [
    {'title':'Toy Story (1995)','rating':4},
    {'title':'Jumanji (1995)','rating':3.5},
    {'title':'Casino (1995)','rating':2},
    {'title':'Othello (1995)','rating':2.5},
    {'title':'Babe (1995)','rating':5}
]

input = pd.DataFrame(user)
input


# In[8]:


inputId = movies[movies['title'].isin(input['title'].tolist())]

input = pd.merge(inputId, input)

input = input.drop('genres',1)

input


# In[9]:


similarUser = ratings[ratings['movieId'].isin(input['movieId'].tolist())]
similarUser.head()


# In[10]:


usersGrp = similarUser.groupby(['userId'])


# In[11]:


usersGrp.get_group(610)


# In[12]:


#Sorts the user subsets according to the highest priority of similarity to the input user
usersGrp = sorted(usersGrp, key=lambda x: len(x[1]), reverse=True)


# In[13]:


#the userId who has watched the most number of common movies with the input user
usersGrp[0]


# In[14]:


#Dataframe of the topmost priority user
usersGrp[0][1]


# # Pearson Correlation for finding similarity

# This function is used to find the similarity between the input user with a subset of common users

# In[15]:


usersGrp = usersGrp[0:200]


# In[16]:


pearsonCoDict = {}

for name, group in usersGrp:
    
    group = group.sort_values(by='movieId')
    input = input.sort_values(by='movieId')
    
    n = len(group)
    
    temp = input[input['movieId'].isin(group['movieId'].tolist())]
    
    rateList = temp['rating'].tolist()
    
    grpList = group['rating'].tolist()
    
    #scipy.stats.pearsonr(rateList, grpList)[0]
    
    Sxx = sum([i**2 for i in rateList]) - pow(sum(rateList),2)/float(n)
    Syy = sum([i**2 for i in grpList]) - pow(sum(grpList),2)/float(n)
    Sxy = sum(i*j for i, j in zip(rateList, grpList)) - sum(rateList)*sum(grpList)/float(n)
    
    if Sxx != 0 and Syy != 0:
        pearsonCoDict[name] = Sxy/np.sqrt(Sxx*Syy)
        
    else:
        pearsonCoDict[name] = 0


# In[17]:


pearsonCoDict.items()


# In[18]:


pearson = pd.DataFrame.from_dict(pearsonCoDict, orient='index')
pearson.head()


# In[19]:


pearson.columns = ['similarity']
pearson['userId'] = pearson.index
pearson.index = range(len(pearson))
pearson.head()


# # Top 50 similar users to input user

# In[20]:


topUsers = pearson.sort_values(by='similarity', ascending = False)[0:50]
topUsers.head()


# In[21]:


topUsersRating = topUsers.merge(ratings, left_on='userId', right_on='userId', how='inner')
topUsersRating.head()


# In[22]:


#Multiplies the similarity by the user's ratings
topUsersRating['weightedRating'] = topUsersRating['similarity']*topUsersRating['rating']
topUsersRating.head()


# In[23]:


#Applies a sum to the topUsers after grouping it up by userId
tempTopUsersRating = topUsersRating.groupby('movieId').sum()[['similarity','weightedRating']]
tempTopUsersRating.columns = ['sum_similarity','sum_weightedRating']
tempTopUsersRating.head()


# In[24]:


#Empty dataframe to store the recommendation score
recommendation = pd.DataFrame()
#Weighted average for calculating the weighted average recommendation score
recommendation['recommendation score'] = tempTopUsersRating['sum_weightedRating']/tempTopUsersRating['sum_similarity']
recommendation['movieId'] = tempTopUsersRating.index
recommendation.head()


# In[25]:


recommendation = recommendation.sort_values(by='recommendation score', ascending=False)
recommendation.head()


# In[26]:


movies.loc[movies['movieId'].isin(recommendation.head(10)['movieId'].tolist())]


# In[29]:


import pickle
pickle.dump(recommendation, open('recommendation.csv','wb'))


# In[ ]:




