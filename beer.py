#!/usr/bin/env python
# coding: utf-8

# # Beer Dataset Challenge
#     presented By: Prerna Gaikwad

# # Content Details
1.Collecting Dataset
2.Initial setup
3.Data loading
4.Data cleaning & preprocessing
5.Solutions
# # 1.Collecting Dataset 
For this exercise, you will be working with beer data which can be downloaded from here 
https://drive.google.com/open?id=1e-kyoB97a5tnE7X4T4Es4FHi4g6Trefq
Unzip the file and you should see a CSV file, called “BeerDataScienceProject.csv”
The columns are as:
1.beer_ABV : Alcohol by volume content of a beer
2.beer_beerId : Unique ID for beer identification
3.beer_brewerId : Unique ID identifying the brewer
4.beer_name : Name of the beer
5.beer_style : Beer Category
6.review_appearance: Rating based on how the beer looks [Range : 1-5]
7.review_palatte : Rating based on how the beer interacts with the palate [Range : 1-5]
8.review_overall : Overall experience of the beer is combined in this rating [Range : 1-5]
9.review_taste : Rating based on how the beer actually tastes [Range : 1-5]
10.review_profileName: Reviewer’s profile name / user ID
11.review_aroma : Rating based on how the beer smells [Range : 1-5]
12.review_text : Review comments/observations in text format
13.review_time : Time in UNIX format when review was recordedQuestions:


    1. Rank top 3 Breweries which produce the strongest beers?
    2. Which year did beers enjoy the highest ratings? 
    3.  Based on the user’s ratings which factors are important among taste, aroma, appearance, and palette?
    4. If you were to recommend 3 beers to your friends based on this data which ones will you recommend?
    5. Which Beer style seems to be the favorite based on reviews written by users? 
    6. How does written review compare to overall review score for the beer styles?
    7. How do find similar beer drinkers by using written reviews only?   

*Please include all plots you created to complete the project and to explain your results. 
# # 2. Initial Setup

# In[4]:


#Importing libraries for Data understanding
import pandas as pd 
import numpy as np
from datetime import datetime


# In[5]:


import warnings
warnings.filterwarnings('ignore')


# In[7]:


#data loading
beer_data=pd.read_csv("BeerDataScienceProject.csv")


# In[8]:


beer_data=data.copy()


# In[9]:


# describe the dataset


# In[10]:


data.head(5)


# In[11]:


data.info()


# In[12]:


data.columns


# In[13]:


data.shape


# In[14]:


# observation:
#Dataset contains 528870 rows and 13 columns


# In[15]:


data.tail(5)


# In[16]:


#Analyzing the Statistical significance of Numeric features
data.describe()


# In[17]:


#Analyzing the Statistical significance of Non-Numeric(categorical)features
data.describe(exclude=np.number)


# In[19]:


#Checking Unique Columns in Dataframe
for col in data.columns:
    if data[col].is_unique:
        print(f'Unique Column : {col}')


# In[20]:


#Re-Setting the Indexes
data=data.reset_index()


# In[21]:


#Checking the Null Counts
data.isnull().sum()


# In[22]:


#Dropping the rows having null values

data=data.dropna()


# In[23]:


data.isnull().sum()


# In[24]:


data.shape


# In[25]:


print('No. of unique values of beer names in the given data :',data.beer_name.nunique(dropna=False))


# In[26]:


print('No. of unique values of beer abv in the given data :',data.beer_ABV.nunique(dropna=False))


# In[27]:


#Removing the duplicated data
#If a user has rated the same beer more than once, then only keep their highest rating and remove others.

data.columns


# In[28]:


data.review_profileName.head(2)


# In[29]:


# Sort by "review_overall" in descending order

data=data.sort_values('review_overall',ascending=False)


# In[30]:


# Keep the highest rating from each "review_profilename" and drop the rest

data=data.drop_duplicates(subset=['review_profileName','beer_beerId'],keep='first')


# In[31]:


#checking shape of non duplicate data. 
data.shape


# In[32]:


#Removing reviews with Ratings <= 0
#Since ratings are on a scale of 1-5, any values in review variables that are less than 1 are not suitable for analysis.
data=data[(data['review_overall']>0)]


# In[33]:


data.shape


# In[34]:


#Univariate and Bivariate analysis of different features


# In[35]:


#beer_name
data['beer_name'].value_counts().head(50).plot.bar(figsize=(16,5),title= 'Most Poular Beers by Name')


# In[36]:


#Beer style

data['beer_style'].value_counts().head(50).plot.bar(figsize=(16,5),title= 'Most Poular Beers by Style')


# In[38]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[40]:


#Beer ABV
plt.figure(figsize=(12,5))
sns.distplot(data['beer_ABV'],bins = 50)
plt.xlabel("Alcohol By Volume")
plt.show()


# In[42]:


#Review Overall

plt.figure(figsize=(16,5))

plt.subplot(121) 
sns.distplot(data.review_overall,bins=50)

plt.subplot(122) 
data['review_overall'].plot.box(title= 'review_overall') 

plt.tight_layout()
plt.show()


# In[43]:


# Plotting Histograms to display the PDF of all the numeric type features in this dataset. 

data.hist(bins = 15,figsize=(16,12))
plt.show()


# # Solution

#Q1.Rank top 3 Breweries which produce the strongest beers?

# In[44]:


#Extracting the relevant columns into new dataframe to use it futher.
abv_data = data[['beer_brewerId','beer_beerId','beer_name','beer_ABV']]


# In[45]:


abv_data.head(5)


# In[46]:


abv_data.shape


# In[47]:


#Applying aggregations over a "beer_ABV" with resoect to "beer_brewerId" and "beer_beerId"
aggrAbvdf = abv_data.groupby(['beer_brewerId','beer_beerId']).agg({'beer_ABV':[np.size,np.mean]})


# In[48]:


aggrAbvdf.head(5)
# Here, the size is corresponds to the number of review logged per unique beer_beerId and
# the mean value tells the mean of the beer_ABV value recorder in the review


# In[49]:


# Now aggregating on a brewery level i.e 'beer_ABV' mean for each brewery
beer_ABVmeanDf = aggrAbvdf.groupby(level='beer_brewerId').mean()
beer_ABVmeanDf.columns = ('mean_size','mean_beer_ABV')


# In[50]:


beer_ABVmeanDf.head(5)


# In[51]:


#dropping the column 'mean_size' and sorting the dataframe in descending order
sorted_beer_ABVmeanDf = beer_ABVmeanDf.drop('mean_size',1).sort_values('mean_beer_ABV',ascending=False).reset_index()


# In[52]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.boxplot(data=sorted_beer_ABVmeanDf['mean_beer_ABV'])


# In[53]:


#Observations :
   # 1. From the above visualization, we can see that the most of the breweries brew around 6% beer_ABV.
    #2. The box plot plot shows the anomalous value of around 25% for beer_brewerId "6513" which is far greater than mean beer_ABV
    #3. We can’t rule out the slight possibility that the actual strongest beer is one with a null value in column "beer_ABV".


# In[54]:


sorted_beer_ABVmeanDf.head(3)


# In[55]:


# observation
#above are the top 3 Breweries producing strongest beers considering anomalous value of 24.69% for brewery "6513"

# Q2 : Which year did beers enjoy the highest ratings ?
# In[56]:


data.head(2)


# In[57]:


data.columns


# In[58]:


#Extracting the relevant columns into new dataframe to use it futher.
review_data=data[['beer_beerId','beer_name','review_overall','review_time']]


# In[59]:


review_data['review_year']=review_data.apply(lambda row: datetime.utcfromtimestamp(row.review_time).strftime("%Y"), axis=1)


# In[60]:


review_data=review_data.reset_index(drop=True)


# In[61]:


review_data.head(5)


# In[62]:


aggrreview_data=review_data.groupby(['review_year'],as_index=False)['review_overall'].agg('mean')


# In[64]:


aggrreview_data.head(5)


# In[65]:


# sorting the dataframe in desceding order


# In[66]:


aggrreview_data=aggrreview_data.sort_values(by=['review_overall'],ascending=False)


# In[67]:


aggrreview_data.head(5)


# In[68]:


import matplotlib.pyplot as plt
fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
ax.bar(aggrreview_data.review_year,aggrreview_data.review_overall)
plt.show()


# In[69]:


#Observation :
   # 1. Observing above bar plot we can say that o year '2000' beer enjoys higest overall ratings of "4.241379"

#Q3 : Based on the user’s ratings which factors are important among taste, aroma, appearance, and palette?
# In[70]:


data.columns


# In[71]:


beerratings_df=data[['beer_beerId','review_appearance','review_palette','review_taste','review_aroma','review_overall']]


# In[72]:


beerratings_df.head(5)


# In[73]:


beerratingsaggr_df=beerratings_df.groupby('beer_beerId').agg(
       {'beer_beerId':'count',
        'review_aroma':'mean',
        'review_taste':'mean',
        'review_appearance':'mean',
        'review_palette':'mean',
        'review_overall':'mean'})
beerratingsaggr_df.columns =('beer_beerId_count','review_aroma','review_taste','review_appearance','review_palette','review_overall')


# In[74]:


beerratingsaggr_df.head(5)


# In[75]:


# Lets observe the correlation between columns.
# Calculate the Pearson's correlation coeff and plot a heatmap.


# In[76]:


corvals=beerratingsaggr_df.corr()


# In[77]:


corvals=corvals.iloc[1:6,1:6]


# In[78]:


corvals


# In[79]:


#masking the upper diagonal matrix
uppermask = np.zeros_like(corvals,dtype=np.bool)


# In[80]:


uppermask[np.triu_indices_from(uppermask,k=1)] = True


# In[82]:


#plotting heatmap
sns.heatmap(corvals,mask=uppermask,annot=True,linewidths=.3)
plt.title('Pearson Correlation coeff')
plt.show()


# In[83]:


#Observations:
   # 1. Observing above heat map we can say that the important factors in deciding overall rating of the beers are aroma(0.88) followeb by taste (0.82) , pallete (0.77) and appearance (0.64)

#Q4 : If you were to recommend 3 beers to your friends based on this data which ones will you recommend?
# In[84]:


data.columns


# In[85]:


#Chosing the relevant features to analyze


# In[86]:


recommedingdata=data[['beer_brewerId','beer_beerId','beer_name','beer_ABV','beer_style','review_overall']]


# In[87]:


# Counting the number of the reviews with respect to each beer


# In[88]:


recommedingdata['review_count']=recommedingdata.groupby(['beer_beerId'])['review_overall'].transform('count')


# In[89]:


recommedingdata.review_count.mean()


# In[90]:


# Plotting Histogram to visualise the distribution review_count 
recommedingdata.review_count.hist(bins=50)
plt.ylabel('Count')
plt.show()


# In[91]:


recommedingdata.shape


# In[92]:


#calculating mean review_overall value for each beer Id 
recommedingdata['review_overall_mean']=recommedingdata.groupby(['beer_beerId'])['review_overall'].transform('mean')


# In[93]:


#Considering 1/4 th of avg review count to filter
filterrecommedingdata=recommedingdata[recommedingdata.review_count >=150]


# In[94]:



aggrfilterrecommedingdata=filterrecommedingdata.groupby('beer_beerId').head(1).sort_values('review_overall_mean',ascending=False)


# In[95]:


# The top 3 suggestions are as below


# In[96]:


aggrfilterrecommedingdata.head(3)

#Q5 : Which Beer style seems to be the favorite based on Reviews written by users? 
#Q6 :How does written review compare to overall review score for the beer style?

# In[97]:


data.columns


# In[98]:


# Considering only relevant columns


# In[99]:


reviewTextData=data[['beer_beerId','beer_name','beer_ABV','beer_style','review_overall','review_text']]


# In[100]:


# lets consider the higher reviews only to calculate the polarity score of review_text and compare it with overall review


# In[102]:


reviewTextData=reviewTextData.loc[reviewTextData['review_overall'] >=4]


# In[103]:


#Resetting Index


# In[104]:


reviewTextData.reset_index(drop=True,inplace=True)


# In[105]:


reviewTextData.head(2)


# In[106]:


# Printing some random reviews to observe the review text


# In[107]:


reviewTextData.review_text[0]


# In[108]:


reviewTextData.review_text[492]


# In[110]:


#Text processing


# In[111]:


import re
#Initial text processing to deconstruct the short forms


# In[112]:


def decontracted(phrase):
    #specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    #general
    phrase = re.sub(r"n\'t","not",phrase)
    phrase = re.sub(r"\'re","are",phrase)
    phrase = re.sub(r"\'s","is", phrase)
    phrase = re.sub(r"\'d","would",phrase)
    phrase = re.sub(r"\'ll","will",phrase)
    phrase = re.sub(r"\'t" ,"not",phrase)
    phrase = re.sub(r"\'ve","have",phrase)
    phrase = re.sub(r"\'m","am",phrase)
    
    return phrase
    


# In[113]:


#Extracting text reciews and applying text pre-processing on it.
from tqdm import tqdm
tqdm.pandas()
preprocessed_reviews = []

for sentence in tqdm(reviewTextData['review_text'].values):
    sentence = decontracted(sentence)
    sentence = re.sub("\S*\d\S*","",sentence).strip()
    preprocessed_reviews.append(sentence)

#Q6.How does written review compare to overall review score for the beer style?
#Q7. How do find similar beer drinkers by using written reviews only?   

# In[115]:


preprocessed_reviews[0]


# In[116]:


#Appending preprocessed reviews to the filtered dataframe

reviewTextData['preprocessed_review_text'] = preprocessed_reviews


# In[117]:


reviewTextData.review_text[0]


# In[118]:


# text analysis

import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# In[119]:


# Instantiating Sentiment Analyzer
sentianalyzer = SentimentIntensityAnalyzer()


# In[120]:


#loop over the 'preprocessed_review_text' column and calculate the polarity score for each review.
reviewTextData['polarity_score2'] = reviewTextData['preprocessed_review_text'].progress_apply(lambda x: sentianalyzer.polarity_scores(x)['compound'])


# In[121]:


# Groupping by the 'beer_beerId' and calculate mean polarity score.
reviewTextDataGroupped=reviewTextData.groupby('beer_style')['polarity_score2'].mean()


# In[122]:


# Lets Sort the groupped data by mean polarity score
reviewTextDataGroupped.sort_values(ascending=False)[0:5]


# In[123]:


# Obsering the top 'polarity_score2' and 'beer_beerId' associated with it
reviewTextData.loc[reviewTextData['beer_style'] == 'Braggot']


# In[124]:


#observation
   # 1. By Observing the mean compund ploarity score , we can say that the beer style "Braggot" is quite most famous.
   # 2. By Observing the mean compund ploarity score calculated we can get an idea how the user written review text is collaborating in calculating the overall review score.


# In[ ]:


# Obsering the top 'polarity_score2' and 'beer_beerId' associated with it
reviewTextData.loc[reviewTextData['beer_'] == 'Braggot'] 


# In[125]:


data1 = reviewTextData.groupby(['beer_name','review_text'])[['review_overall']].sum().sort_values('review_overall',ascending = False).reset_index()
data1


# In[126]:


data1.loc[data1.review_text != '#NAME?'].head(5)


# In[127]:


#Observation:
# Black tuesday,Cornballer beer_name mostly used by drinkers based on written reviews

