
**EDA Preprocessing visualization`**

---
# Roll: 20231141
**Name: Mohammad Kabir Hosen**
Batch-11


# #Data Visualization, Data Preprocessing and Exploratory Data Analysis
# Choose one cars dataset (Audi, BMW, Ford, Hyundai, Skoda, VW) for the tasks (1 – 10):
# https://www.kaggle.com/datasets/aishwaryamuthukumar/cars-dataset-audi-bmw-ford-hyundai-skoda-vw
    
# 1. Create a Pie Chart and a barplot for any categorical variable. Compare which plot is better and why?
# Use markdown cells to write your explanations.

import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns

df = pd.read_csv("C:\\Users\\User\\Downloads/bmw - bmw.csv")
df


# print(type(df))
# display(df.head())
# display(df.tail())
# df.info()
df.dtypes
# df.sample(5,random_state = 4)
# df.columns
# df.index
df.shape

print('Number of Rows = ', df.shape[0])
print('Number of Columns = ', df.shape[1])


# df.info()
df.isnull().sum()
df.notnull().sum()


df['model'].unique()

df['fuelType'].value_counts()

# print(df['model'].unique())
print(type(df['model'].unique()))
len(df['model'].unique())

df['transmission'].value_counts().plot.pie(autopct = '%.2f%%')

print(df["fuelType"].value_counts())
print(type(df["fuelType"].value_counts()))
df_fuelType = pd.DataFrame(df["fuelType"].value_counts())
display(df_fuelType.head())
print(df_fuelType.index)
print(df_fuelType.columns)

df_fuelType = pd.DataFrame(df['fuelType'].value_counts())
df_fuelType = df_fuelType.reset_index()
df_fuelType = df_fuelType.rename(columns = {'index':'fuelType',
'fuelType':'no_of_cars'})
df_fuelType['% of cars'] = (df_fuelType['no_of_cars']/df.shape[0])*100

# df_fuelType['no_of_cars'].sum()
# df.shape[0]

df_fuelType

sns.barplot(x="fuelType",
y="% of cars",
data=df_fuelType,
color="red",
alpha=0.8)
plt.xlabel("Types of fuel")
plt.ylabel("% of cars")
plt.title("Percentage of cars present in each fuelType")
plt.yticks(np.arange(0,101,10))
plt.show()


df[["year","price"]].head


sns.set_style('darkgrid')


sns.barplot(x= 'price', y= 'mpg' , data=df);


sns.barplot(x= 'price', y= 'model' , data=df);


#  Bar plot is better.
#  * Bar plots provide a clear visual representation of the data, allowing for easy comparison between categories.
# 
#  * pie charts can be less intuitive for comparing values since the angles and areas of the slices are more challenging to accurately perceive.


2. Create Two scatterplots with numeric columns.


# In[14]:


import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns


df = pd.read_csv("C:\\Users\\User\\Desktop\\New folder/bmw - bmw.csv")
df


sns.set_style("darkgrid")


sns.scatterplot(df['year'],df['price'] );

sns.scatterplot(df['year'],df['transmission'] );

sns.scatterplot(df['year'],df['price'] );


df[["year","price","transmission"]].head(10)

df[["year","price","transmission"]].head(10)


# 3. Create Two regression plots with numeric columns.


sns.regplot(df['year'],df['price'] )


sns.regplot(df['year'],df['mileage'] )


sns.set_style("darkgrid")


sns.scatterplot(df['model'],df['price'] , hue= df['transmission']);


# 4. Create a Pair plot with numeric columns.


sns.pairplot(df);


sns.pairplot(df, corner=True, hue="transmission")


#8. Create boxplot or violin plot (one or more) and histogram (one or more) with the numeric columns.
Add a categorical column as x-axis or hue if you think it's important. Explain which chart is better
get_ipython().set_next_input('boxplot or violinplot');get_ipython().run_line_magic('pinfo', 'violinplot')


sns.boxplot(df['year'],df['price'] )

sns.boxplot(df['year'],df['price'] , hue= df['transmission'])


sns.boxplot(df['year'],df['model'] )


sns.violinplot(df['year'])


sns.violinplot(df['year'],df['mileage'] )


sns.violinplot(df['model'],df['price'] , hue= df['transmission']);


sns.violinplot(df['year'],df['price'] )

sns.violinplot(df['model'],df['price'] , hue= df['transmission']);


sns.histplot(df['year'], kde=True )


# Both boxplots and violin plots are commonly used for visualizing the distribution of numerical data. While they have some similarities, there are also differences in terms of their characteristics and the insights they provide. Whether one chart is better than the other depends on the specific data and the purpose of the analysis. Let's explore each type in more detail:
# 
# Boxplot:
# A boxplot, also known as a box-and-whisker plot, displays the summary statistics of a dataset, including the minimum, first quartile (Q1), median (second quartile, Q2), third quartile (Q3), and maximum. The box represents the interquartile range (IQR), which is the range between Q1 and Q3. A horizontal line inside the box represents the median. The whiskers extend from the box to the minimum and maximum values within a specified range, typically 1.5 times the IQR. Points outside this range are considered outliers and shown as individual data points.
# 
# Boxplots are useful for:
# 
#     Identifying the range and distribution of the data.
#     Detecting skewness and outliers.
#     Comparing distributions across different categories or groups.
# 
# Violinplot:
# A violin plot combines aspects of a boxplot and a kernel density plot. It provides a more detailed representation of the data distribution by displaying a rotated kernel density plot on each side of a central box. The width of the violin at any given point represents the density or frequency of data values at that point. The plot can be mirrored, or split for side-by-side comparison of different groups.
# 
# Violinplots are useful for:
# 
#     Visualizing the shape and distribution of the data, including multimodality.
#     Comparing distributions across different categories or groups.
#     Combining the benefits of a boxplot and a density plot.
# 
# Choosing between boxplots and violin plots depends on several factors:
# 
#     Data characteristics: If the focus is on summary statistics and identifying outliers, a boxplot may be more appropriate. If the shape and multimodality of the distribution are important, a violin plot provides a richer representation.
#     Audience and context: Consider the familiarity of the audience with each chart type and the context of the analysis. Boxplots are more widely recognized and often preferred for simplicity. Violin plots may be more suitable when a more detailed representation is necessary or when the audience is familiar with the plot type.
#     Specific insights: Consider the specific insights you want to extract from the visualization. If you primarily want to compare distributions across groups, both boxplots and violin plots can be effective, but violin plots offer more information on the shape of the distributions.
# 
# In conclusion, both boxplots and violin plots have their strengths and should be chosen based on the specific requirements of the analysis, data characteristics, and the intended audience.


df[["year","price","transmission"]].head(10)

sns.histplot(df['year'], kde=True)

df[["year","price","transmission"]].head(10)


sns.histplot(df.price, binwidth = 5, stat = 'probability')


10. Perform Exploratory Data Analysis (EDA) using groupby/pivot_table and barplot (total 9 barplots),
based on model, transmission, and fuelType,
get_ipython().set_next_input('a) What are the top 5 selling car models/transmission/fuelType in the dataset');get_ipython().run_line_magic('pinfo', 'dataset')
get_ipython().set_next_input("b) What's the average selling price of the top 5 selling car models/transmission/fuelType");get_ipython().run_line_magic('pinfo', 'fuelType')
get_ipython().set_next_input("c) What's the total sale of the top 5 selling car models/transmission/fuelType");get_ipython().run_line_magic('pinfo', 'fuelType')



# Get the top 5 selling car models
top_models = pd.pivot_table(df, index='model', aggfunc='size').sort_values(ascending=False).head(5)

# Get the top 5 selling transmission types
top_transmissions = pd.pivot_table(df, index='transmission', aggfunc='size').sort_values(ascending=False).head(5)

# Get the top 5 selling fuel types
top_fuel_types = pd.pivot_table(df, index='fuelType', aggfunc='size').sort_values(ascending=False).head(5)

# Print the results
print("Top 5 Selling Car Models:")
print(top_models)
print("\nTop 5 Selling Transmission Types:")
print(top_transmissions)
print("\nTop 5 Selling Fuel Types:")
print(top_fuel_types)

# Create a bar plot from the top 5 models
sns.set_style('darkgrid')
sns.barplot(top_models.index, top_models.values)
plt.xlabel('Model')
plt.ylabel('Count')
plt.title('Top 5 Models')
plt.xticks(rotation=45)

# Display the bar plot
plt.show()


# Create a pivot table based on 'model', 'transmission', and 'fuelType'
pivot_table = pd.pivot_table(df, index=['model', 'transmission', 'fuelType'], aggfunc='size')

# Sort the pivot table in descending order and select the top 5
top_5 = pivot_table.sort_values(ascending=False).head(5)

# Print the top 5 results
print("Top 5 combinations of model, transmission, and fuelType:")
print(top_5)


# Get the top 5 selling car models
top_models = pd.pivot_table(df, index='model', aggfunc='size').sort_values(ascending=False).head(5)

# Get the top 5 selling transmission types
top_transmissions = pd.pivot_table(df, index='transmission', aggfunc='size').sort_values(ascending=False).head(5)

# Get the top 5 selling fuel types
top_fuel_types = pd.pivot_table(df, index='fuelType', aggfunc='size').sort_values(ascending=False).head(5)

# Print the results
print("Top 5 Selling Car Models:")
print(top_models)
print("\nTop 5 Selling Transmission Types:")
print(top_transmissions)
print("\nTop 5 Selling Fuel Types:")
print(top_fuel_types)

# Create a bar plot from the top 5 models
sns.barplot(top_models.index, top_models.values)
plt.xlabel('Model')
plt.ylabel('Count')
plt.title('Top 5 Models')
plt.xticks(rotation=45)

# Display the bar plot
plt.show()


# Create a pivot table based on 'model', 'transmission', and 'fuelType'
pivot_table = pd.pivot_table(df, index=['model', 'transmission', 'fuelType'], aggfunc='size')

# Sort the pivot table in descending order and select the top 5
top_5 = pivot_table.sort_values(ascending=False).head(5)

# Print the top 5 results
print("Top 5 combinations of model, transmission, and fuelType:")
print(top_5)


# Get the top 5 selling car models
top_models = pd.pivot_table(df, index='model', aggfunc='size').sort_values(ascending=False).head(5)

# Get the top 5 selling transmission types
top_transmissions = pd.pivot_table(df, index='transmission', aggfunc='size').sort_values(ascending=False).head(5)

# Get the top 5 selling fuel types
top_fuel_types = pd.pivot_table(df, index='fuelType', aggfunc='size').sort_values(ascending=False).head(5)

# Print the results
print("Top 5 Selling Car Models:")
print(top_models)
print("\nTop 5 Selling Transmission Types:")
print(top_transmissions)
print("\nTop 5 Selling Fuel Types:")
print(top_fuel_types)

# Create a bar plot from the top 5 models
sns.barplot(top_models.index, top_models.values)
plt.xlabel('Model')
plt.ylabel('Count')
plt.title('Top 5 Models')
plt.xticks(rotation=45)

# Display the bar plot
plt.show()


# Create a pivot table based on 'model', 'transmission', and 'fuelType'
pivot_table = pd.pivot_table(df, index=['model', 'transmission', 'fuelType'], aggfunc='size')

# Sort the pivot table in descending order and select the top 5
top_5 = pivot_table.sort_values(ascending=False).head(5)

# Print the top 5 results
print("Top 5 combinations of model, transmission, and fuelType:")
print(top_5)



Import the ‘ODI_cricket.xlsx’ file ‘bowler’ sheet in your notebook for the tasks (11 – 26)
Actual data source: https://stats.espncricinfo.com/ci/content/records/283193.html
11. Display the first 10 rows and last 3 rows of the dataframe.


11. Display the first 10 rows and last 3 rows of the dataframe.


import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns

df = pd.read_csv("C:\\Users\\User\\Desktop\\Media/ODI_bowler.csv")
df
first_10_rows = df.head(10)
last_3_rows = df.tail(3)
print("First 10 rows:")
print(first_10_rows)
print("Last 3 rows:")
print(last_3_rows)


df.tail()


# # 12. Create a markdown cell and explain the meaning of each column.
# 
# ## Column Meanings
# 
# 1. **Match_ID:** An identifier for each cricket match in the dataset.
# 
# 2. **Date:** The date when the match took place
# 
# 3. **Team_1:** The name of the first team participating in the match.
# 
# 4. **Team_2:** The name of the second team participating in the match.
# 
# 5. **Venue: **The location or stadium where the match was played.
# 
# 6. **City:** The city where the match was played.

# 14. find the data statistics and check for the data types.


data_statistics = df.describe()
data_types = df.dtypes
print("Data statistics:")
print(data_statistics)
print("Data types:")
print(data_types)


get_ipython().set_next_input('15. Are there any missing values present in the dataset');get_ipython().run_line_magic('pinfo', 'dataset')


df.isnull().sum()

df = pd.read_csv("C:\\Users\\User\\Desktop\\Media/ODI_bowler.csv")
df

num_rows, num_columns = df.shape
print("Number of rows:", num_rows)
print("Number of columns:", num_columns)


16. Rename the column names appropriately.

df = pd.read_csv("C:\\Users\\User\\Desktop\\Media/ODI_bowler.csv")
new_column_names = {
    'Balls': 'Maximum Balls',
    'Wickets': 'Maximum Wicets',
    'Mats': 'New Mats',}

df = df.rename(columns=new_column_names)


print(df.rename)


# 17. How many players played for ICC?

df = pd.read_csv("C:\\Users\\User\\Desktop\\Media/ODI_bowler.csv")
df


icc_players = df[df['Player'].str.contains('ICC', case=False)]
num_icc_players = len(icc_players)

print("Number of players who played for ICC:", num_icc_players)


# 18. How many different countries are present in this dataset?

df = pd.read_csv("C:\\Users\\User\\Desktop\\Media/ODI_bowler.csv")

# Extract country abbreviations from the Player column
#countries = df['Player'].apply(lambda x: re.findall(r'\(([A-Z]+)\)', x)[0])
countries = df['Player'].str.extract('\((.*?)\)')[0].unique()

# Count the number of unique countries
#num_countries = len(countries.unique())

print("Number of different countries: ", countries)


# 19. Which player(s) had played for the longest period of time?

# Extract the start and end years from the Span column
df['Start Year'] = pd.to_numeric(df['Span'].str.split('-').str[0])
df['End Year'] = pd.to_numeric(df['Span'].str.split('-').str[1])

# Calculate the career duration for each player
df['Career Duration'] = df['End Year'] - df['Start Year']

# Find the maximum career duration
max_duration = df['Career Duration'].max()

# Identify the player(s) with the longest duration
longest_career_players = df[df['Career Duration'] == max_duration]['Player'].tolist()

print("Player(s) with the longest period of time: ")
for player in longest_career_players:
    print(player)



get_ipython().set_next_input('20. Which player(s) had played for the shortest period of time');get_ipython().run_line_magic('pinfo', 'time')


# Find the minimum career duration
min_duration = df['Career Duration'].min()

# Identify the player(s) with the shortest duration
shortest_career_players = df[df['Career Duration'] == min_duration]['Player'].tolist()

print("Player(s) with the shortest period of time: ")
for player in shortest_career_players:
    print(player)


get_ipython().set_next_input('21. How many Australian Bowlers are present in this dataset');get_ipython().run_line_magic('pinfo', 'dataset')

# Count the number of Australian bowlers
num_australian_bowlers = sum(countries == 'AUS')

print("Number of Australian bowlers: ", num_australian_bowlers)


get_ipython().set_next_input('22. Is there any Bangladeshi player present in this dataset');get_ipython().run_line_magic('pinfo', 'dataset')


# Check if there is any Bangladeshi player
bangladeshi_player_present = any(countries == 'BAN')

if bangladeshi_player_present:
    print("Yes, there is at least one Bangladeshi player in the dataset.")
else:
    print("No, there are no Bangladeshi players in the dataset.")



sorted_by_economy = df.sort_values('Econ', ascending=True)
print(sorted_by_economy[['Player', 'Econ']].head(1))


get_ipython().set_next_input('23. Which player had the lowest economy rate');get_ipython().run_line_magic('pinfo', 'rate')


sorted_by_strike_rate = df.sort_values('SR', ascending=True)
print(sorted_by_strike_rate[['Player', 'SR']].head(1))


get_ipython().set_next_input('Which player had the lowest bowling average');get_ipython().run_line_magic('pinfo', 'average')


sorted_by_bowling_average = df.sort_values('Ave', ascending=True)
print(sorted_by_bowling_average[['Player', 'Ave']].head(1))


# 26. Remove Unnecessary columns if needed.


columns_to_drop = ['BBI', 'BBM', '10']  # Specify the column names you want to remove
df = df.drop(columns_to_drop, axis=1)

print(df)



