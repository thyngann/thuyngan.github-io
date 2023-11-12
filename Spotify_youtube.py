import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#reading the data
df=pd.read_csv(r"C:\Users\Admin\Downloads\spotify_Youtube.csv")
df.describe()
df.info()#get the information of dataset
df.shape
df.columns
df.nunique()
df.isna().sum()
df.dropna(inplace=True)
df.isnull().sum()
df.shape

df['Views']=df['Views'].astype('int')
df['Likes']=df['Likes'].astype('int')
df['comment']=df['Comments'].astype('int')

#processing the Data
df.head(10)
#determining the top 10 artists on the Spotify platform whose songs have more views
top_artists=df.groupby(['Artist'])['Views'].sum().reset_index().sort_values(by='Views', ascending=False).head(10)
top_artists.set_index('Artist', inplace=True)
top_artists.plot(kind='bar', color='g')
plt.xlabel('Artist')
plt.ylabel('Views')
plt.title('Top 10 Artist with higher views')
plt.xticks(rotation=70)
plt.show()
#groupby the artist

#make graph of the top 10 most liked artists in Youtube, bar chart
top_likes = df.groupby('Artist')['Likes'].sum().sort_values(ascending=False).head(10)
sns.barplot(data=top_likes.reset_index(),x='Likes',y='Artist')
plt.title('Top 10 most Liked Artist')
plt.show()

#Determine the top 10 Streamers on Spotify

ax = df.groupby('Artist').agg({'Likes': 'sum', 'Stream': 'sum'}).reset_index()
bx = ax.sort_values('Likes', ascending=False).head(10)
cx = ax.sort_values('Stream', ascending=False).head(10)
df.groupby('Artist')['Stream'].agg('sum').head(10)
sns.barplot(data=cx, y='Stream', x='Artist')
plt.xticks(rotation=80)
plt.title('Top 10 streamers')
plt.show()

#Who is the most commented artist on Youtube
ax=df.groupby('Artist')['Comments'].sum().reset_index().sort_values(by='Comments',ascending=False).head(10)
sns.barplot(data=ax,y='Artist',x='Comments')
plt.title('Top 10 most commented Artist')
plt.xlabel('Comment Count') #Add an x-axis label
plt.ylabel('Artist')
plt.show()
# as we can see BTS on the top first position with 39 million comments after this blackpink 19 million ,stray kids, twice,psy from third to fifth position with 8.4 million to 7.4 million comments
#What are top 10 songs on youtube with higher views

tc=df.groupby('Track').agg({'Views':'sum','Comments':'sum','Likes':'sum'}).reset_index()
bv=tc.sort_values('Views',ascending=False).head(10)
plt.barh(bv['Track'],bv['Views'],color='g')
plt.title('Top 10 most Viewed tracks')
plt.show()
# as we can see most viewed song on youtube is Swalla .feat. Nicki Minaj & Ty Dolla $ign with 5.1 billion views and on second Thrift shop with 4.5 billion views
#  on third ,fourth and fifth something just like this,sin pijama and somebody that i used with 4.1 to 4.2 billion views

#determine most commented song on youtube
av=tc.sort_values('Comments',ascending=False).head(10)
plt.barh(av['Track'],av['Comments'])
plt.title('Top 10 most commented Tracks')
plt.show

#Most liked song on Youtube or Sportify
sn=tc.sort_values('Likes',ascending=False).head(10)
plt.barh(sn['Track'],sn['Likes'],color='r')
plt.title('Top 10 ,ost liked tracks')
plt.show()

#most viewed song relationship with stream
top_songs=df.groupby('Track').agg({'Views':'sum','Stream':'sum'}).reset_index().sort_values(by='Views',ascending=False).head(10)
#which album type is getting more views
ac=df.groupby(['Album_type'])['Views'].sum().head()
gp=df.groupby('Album_type')['Views'].sum().reset_index()
plt.figure(figsize=(8, 6)) #Set the figure size
sns.set(style="whitegrid")
plt.pie(data=gp,x='Views',labels='Album_type',autopct='%1.0f%%')
plt.title('Album Views by Types')
plt.show()
# #### From the above Pie chart we can make few conclusions like:
# * album type is most popular type ,the data shows in album type has 1.148762e+12 views which is 75% of data total data, and it is higher compared to other compilation and single types
#Top 10 licensed album on youtube & sportify
al=df[df['Licensed']==1]
la=al['Album'].value_counts()[:10]
plt.pie(la,autopct='%.2f%%', labels=la.index)
plt.show()

#make a histogram to visualize all the important factors of singers
df.hist(bins=40,figsize=(10,18))
plt.figure(figsize=(10, 18))
plt.tight_layout()
plt.show()
#Display heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), linewidth=0.6, annot=True, fmt='.1f',linecolor='black', cmap="jet")
plt.axis('tight')
plt.title('Factor Correlation')
plt.show()

#determine the relationship between tempo and speechiness
fig=plt.figure(figsize=(10, 5))
sns.jointplot(data=df, x='Tempo', y='Speechiness', kind='Scatter')
plt.ylabel("Speechiness", labelpad=20)
plt.xlabel("Tempo", labelpad=20)
plt.axis('tight')
plt.show()

