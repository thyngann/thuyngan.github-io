import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import geopandas as gpd
import warnings

pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=DeprecationWarning)

df = pd.read_csv('netflix_titles.csv')
df.head(2)

print('before cleaning NaN/duplicated: \n')
print('row:', df.shape[0])
print('columns:', df.shape[1])

df.info()

dupe = df.duplicated().sum()
print('duplicated entries:', dupe)

missing = df.isna().sum()
print('Missing values: \n')
print(missing[missing>0])

df['director'].fillna('Unknown',inplace = True)
df['cast'].fillna('Unknown',inplace = True)
df['country'].fillna('Unknown',inplace = True)
df.dropna(subset=['date_added','rating','duration'],inplace = True)

missing_after = df.isna().sum()
print('Missing values after cleaning: \n')
print(missing_after)

print('after cleaning NaN/duplicated: \n')
print('row:', df.shape[0])

df['description_length'] = df['description'].str.len()
df.head(2)

df.shape

df.nunique()

df.columns

df = df[['type', 'director', 'cast', 'country', 'date_added', 'release_year', 'rating', 'duration', 'listed_in', 'description_length']]
df.head(2)

df.info()

df['date_added'] = df['date_added'].str.strip()
df['date_added'] = pd.to_datetime(df['date_added'],format = '%B %d, %Y')
df.info()

df.head(2)

df['added_date'] = df['date_added'].dt.day
df['added_month'] = df['date_added'].dt.month
df['added_year'] = df['date_added'].dt.year
df.nunique()

#GroupBy Type
df_type = df.groupby(by = 'type').agg(count = ('type','count')).reset_index()

#Identify highest values
most_type = df_type[df_type['count'] == df_type['count'].max()]['type'].values[0]
most_type_value = df_type[df_type['count'] == df_type['count'].max()]['count'].values[0]
most_type_pct = (most_type_value / df_type.sum().iloc[1]) * 100

#plot chart
sns.barplot(data=df_type, x='type', y='count')
plt.xlabel('type')
plt.ylabel('No.of shows')
plt.title('No.of shows by types')
plt.show()

#comment
print(f'Most of the shows on Netflix are {most_type} with {most_type_value} shows, contains about {most_type_pct:.2f}')

#Filter 2 specific dataframe to see the duration of each
df_movie = df[df['type'] =='Movie']
df_tvshow = df[df['type'] == 'TV Show']

#Convert dtype
df_movie['duration'] = df_movie['duration'].str.replace(' min', '')
df_movie['duration'] = df_movie['duration'].astype(int)

df_tvshow['duration'] = df_tvshow['duration'].str.replace(' Season', '').str.replace('s', '')
df_tvshow['duration'] = df_tvshow['duration'].astype(int)

#Groupby Duration
df_movie_duration = df_movie.groupby(by = 'duration').agg(count = ('duration', 'count')).reset_index()
df_tvshow_duration = df_tvshow.groupby(by ='duration').agg(count = ('duration', 'count')).reset_index()

fix,axes = plt.subplots(1,2,figsize = (12,4))

sns.kdeplot(data=df_movie['duration']
            ,fill=True
            ,ax=axes[0])
axes[0].set_xlabel('Duration (mins)')
axes[0].set_ylabel('No.of shows')
axes[0].set_title('No.of Movies')

sns.kdeplot(data=df_tvshow['duration']
            ,fill=True
            ,ax = axes[1])
axes[1].set_xlabel('Duration (Seasons)')
axes[1].set_ylabel('No. of shows')
axes[1].set_title('No.of TV Shows')

plt.suptitle('Duration distribution of 2 type of shows on Netflix')
plt.tight_layout()
plt.show()

#comment
print('A good number of movies on Netflix are among the duration of 75-120 mins.\n'
      'It is acceptable considering the fact that a fair amount of the users cannot\n'
      'watch a movie in a too long time.\n'
      '\n'
      'The fluctuation in the number of Seasons of TV shows on Netflix typically ranges from 1 to 2.\n'
      'This reflects the stability and consistency in content delivery and is considered an acceptable\n'
      'average figure in the entertainment industry.\n')

#GroupBy Rating
df_rating = df.groupby(by='rating').agg(counting=('rating', 'count')).reset_index().sort_values(by='counting', ascending=False)

#Identify most popular rating
most_popular_rating = df_rating[df_rating['counting'] == df_rating['counting'].max()]['rating'].values[0]
most_popular_rating_value = df_rating[df_rating['counting'] == df_rating['counting'].max()]['counting'].values[0]

plt.figure(figsize=(10, 6))
#Rating distribution
sns.barplot(data=df_rating, x='rating', y='counting')
plt.xlabel('Rating')
plt.ylabel('No. of shows')
plt.title('Rating distribution')
plt.show()

#Comment
print(f"Most popular rating: {most_popular_rating}")
print(f"Total no. of shows: {most_popular_rating_value}")

#which is the "boom" year of Netflix?

#groupby added_year
df_year = df.groupby(by = 'added_year').agg(counting = ('added_year', 'count')).reset_index().sort_values(by = 'added_year')
#Identify max/min count
max_count = max(df_year["counting"])
min_count = min(df_year["counting"])
max_year = df_year["added_year"][df_year["counting"].idxmax()]
min_year = df_year["added_year"][df_year["counting"].idxmin()]

#plot chart
sns.lineplot(data = df_year
             ,x = 'added_year'
             ,y = 'counting')
plt.scatter(max_year, max_count, color='red', zorder=5)
plt.scatter(min_year, min_count, color='red', zorder=5, marker='o')
plt.text(max_year - 0.3, max_count + 50, f'Max = {max_count}', ha ='right')
plt.text(min_year, min_count + 50, f'Min = {min_count}', ha='right')

plt.axvline(x=max_year, color='gray', linestyle='--', label=f'Max Year({max_year})')

plt.xticks(df_year["added_year"])
plt.ylim(0,2500)
plt.xlabel("Added Year")
plt.title("No. of shows by Year")
plt.xticks(rotation=45)

plt.show()

#comment
print(f"It can be clearly seen that the No. of shows witnessed the lowest value at {min_count} in {min_year}.\n"
      f"This number dramatically multiplied {max_count/min_count:.2f} times by the end of {max_year}.\n"
      "\n"
      "In 2019, due to the effect of the Covid-19 epidemic, with people in many parts of the world restricted from their homes "
      "and social activities, many users have increased their demand for online entertainment services. Netflix has benefited "
      "from this increase in demand for online movies and home entertainment.")

#Across each year, is there any different between months?
#Create matrix for heatmap
release_heatmap = df.pivot_table(values ='type', index = 'added_month', columns='added_year', aggfunc ='count')
#plot.heatmap
plt.figure(figsize=(10, 5))

sns.heatmap(data = release_heatmap, cmap='YlGnBu'
 , xticklabels=release_heatmap.columns
 , yticklabels=release_heatmap.index
 , square=False
 , cbar_kws={'orientation': 'horizontal'}
 , linewidths= 0.5
 , linecolor='black'
 )
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.xlabel('Year')
plt.ylabel('Month')
plt.title('Heatmap of added shows by year and month')
plt.show()

#Comment
print(\
    'If the highest year - 2019 is considered, May, August and September were \n\
      the months when much fewer shows was on-air compared to the other months.\n\
      \n\
      Therefore,selecting these months could increase the chances of success fora new show release on Netflix.'
)

#What are the top 10 directors having highest number of shows?
#Identify top 10 directors
df_director = df.groupby(by ='director').agg(count=('director','count')).reset_index().sort_values(by ='count')
df_top10_director = df_director[df_director['director']!='Unknown'].reset_index(drop = True).head(10)
#plot chart
sns.barplot(data=df_top10_director, y='director', x='count')
plt.xlabel('No. of shows')
plt.ylabel('Director')
plt.title('Top 10 directors by No. of Shows')
plt.show()

#top 5 director name
director_name = ",".join(director_name for director_name in df_top10_director['director'].head(5))
df_top10_director['count_text'] = df_top10_director['count'].astype(str)
director_value = ",".join(director_value for director_value in df_top10_director['count_text'].head(5))

#comment
print("We can see that there were 5 directors who contributed the highest number of shows.")
print(f"They are {director_name} with the number of shows being {director_value}, respectively.")

#groupBy Country
df_country = df.groupby(by='country').agg(count=('country', 'count')).reset_index().sort_values(by ='count',ascending=False)
most_country = df_country[df_country['count'] == df_country['count'].max()]['country'].values[0]
most_country_value = df_country[df_country['count'] == df_country['count'].max()]['count'].values[0]

#plot map
shapemap = gpd.datasets.get_path('naturalearth_lowres')
world = gpd.read_file(shapemap)
merged = world.set_index('name').join(pd.DataFrame(df_country).set_index('country'))
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
merged.plot(column='count', cmap='Reds', linewidth=0.5, ax=ax, edgecolor='0.7', legend=True, legend_kwds={'orientation': "horizontal", 'shrink': 0.7})
ax.set_title('No. of shows by country')

plt.tight_layout()
plt.show()
#comment
print(f'It can be seen that, the darkest color is {most_country} with the number of shows are {most_country_value}.')
