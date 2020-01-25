# # Exploratory Data Analysis on Zomato Data Set

# In[78]:

import pandas as pd
df = pd.read_csv('data/zomato.csv', encoding = 'latin-1',  sep=',')
df.head()


# ### Let's Visualize the % of data from different countries

# In[79]:
print("ALL COUNTRIES CODE=")

countries = df['Country Code'].unique()
print(countries)
arr = []
for c in countries:
    dframe = df[df['Country Code']==c]
    n = dframe['Country Code'].count()
    arr.append(n)
print(arr)

import matplotlib.pyplot as plt
plt.pie(arr, labels = countries, autopct='%.0f%%', radius = 2)
plt.show()


# We Can see from the above figure that 91% of the data comes from the country code 1 (India)
# Hence, we can take only the data from India for our Analysis.

dfIndia = df[df['Country Code']==1]
dfIndia = dfIndia.reset_index(drop=True) #resetting the indices
dfIndia.head(n=3)    ##NUMber of Rows to be returned


# ### Let's clean up a few fields of the Data
dfIndia.isnull().values.any() # Checking for Null Values


# There are no null values in the Data
print("UNIQUE VALUE IN PERTICULAR COULUMN")
print(dfIndia['Has Table booking'].unique())
print(dfIndia['Has Online delivery'].unique())
print(dfIndia['Is delivering now'].unique())
print(dfIndia['Rating text'].unique())


# Encoding the Data and dropping the 'Not rated rows'

cleanup = {'Has Table booking': {'Yes': 1, 'No': 0}, #Encoding Yes as 1 and No as 0
           'Has Online delivery': {'Yes': 1, 'No': 0},
           'Is delivering now' : {'Yes': 1, 'No': 0},
           'Rating text' : {'Not rated': 0, 'Poor': 1, 'Average': 2, 'Good': 3, 'Very Good': 4, 'Excellent' : 5}}
dfIndia.replace(cleanup, inplace = True)
noRatng = dfIndia[dfIndia['Rating text']==0]
print("NON RATED Rows=")
print(noRatng['Rating text'].count())
dfIndia = dfIndia[dfIndia['Rating text']!=0]
dfIndia.head()


# Calculating the number of 0s in the column, Avg. Cost of two and replacing it with the mean


totalzero = (dfIndia['Average Cost for two']== 0).sum()
#print(totalzero)
n_sum = dfIndia['Average Cost for two'].sum()
n_total = dfIndia['Average Cost for two'].count()
print("AVERAGE COST OF TWO PERSON")
print(n_sum/n_total)
cleanzero = {'Average Cost for two': {0: 700}}
dfIndia.replace(cleanzero, inplace = True)


# ### Let us now Visualize the graphs of different fields and check the relationships

# 1. Different Cities of India in the Data Set

# In[85]:
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# We can see that the city of New Delhi  DOMINATED the data set

# 2. Will the number of votes influence the Rating?

#GRAPH FOR AGGREGATE RATING VS VOTES

plt.style.use('ggplot')
a = dfIndia['Aggregate rating']
b = dfIndia['Votes']
plt.xlabel('Rating')
plt.ylabel('Number of Votes')
plt.title('Rating vs No. of Votes')
plt.bar(a,b, width = 0.2, linewidth = 10, linestyle = 'solid')
plt.show()


# From the graph, it's highly evident that, higher the number of votes, higher is the rating.

# 3. Will the average cost for two influence the rating?

#GRAPG IS RATING VS AVERAGE COST OF 2

plt.style.use('ggplot')
x = dfIndia['Aggregate rating']
y = dfIndia['Average Cost for two']
plt.xlabel('Rating')
plt.ylabel('Average Cost for two(In Indian Rupees)')
plt.title('Rating vs Cost for Two')
plt.bar(x,y, width = 0.2, linewidth = 10, linestyle = 'solid', color = 'green')
plt.show()


# We can see from the above graph that as the average cost for two increases, the ratings go higher. But it decreases after the Rating of 4.0.
# Hence, we can say that there are a few high rated restaurants with lesser average cost for two.

# ## From the Dataset, we can see that the 'Cuisines' Column has multiple values.
# ### We can take only the Main cuisine of the restaurant and do One-hot encoding on the different types of cusines.

# NEW COULUM CALLED AS DUMMY VARIABLE and that categoricaldata =Nominal varialbles
dftemp = pd.concat([dfIndia, dfIndia['Cuisines'].str.split(',',expand=True)], axis = 1) #Expanding the different values in cuisines
dftemp = dftemp.rename(columns={0:'Cuisine 1'}) #Renaming the Main Cusine as Cuisine 1

# DUMMY VARIABLE TRAP= Multicolinear problem of trap  we are dropping bacause it mess Us all the dataset
dftemp = dftemp.drop(1,axis =1) # Dropping all other Cuisine Values
dftemp = dftemp.drop(2,axis =1)
dftemp = dftemp.drop(3,axis =1)
dftemp = dftemp.drop(4,axis =1)
dftemp = dftemp.drop(5,axis =1)
dftemp = dftemp.drop(6,axis =1)
dftemp = dftemp.drop(7,axis =1)

dfClean = pd.get_dummies(dftemp, columns = ['Cuisine 1']) #One-hot encoding
dfClean.head()


# Dropping all the Unwanted Columns. Reordering the columns and saving it to a new CSV file.

# DISCARDING THE COULUMS WHICH IS NOTREQUIRED FOR FURTHER PROCESS.
#Lable Encoding to one hot Encoding

dfCleanIndia = dfClean.drop(['Cuisines', 'Country Code', 'Rating color', 'Switch to order menu', 'Currency', 'Address', 'Locality', 'Locality Verbose'],  axis =1)
cols = list(dfCleanIndia.columns.values)
cols.pop(cols.index('Rating text'))
cols.pop(cols.index('Aggregate rating'))
dfCleanIndia = dfCleanIndia[cols+['Rating text']+['Aggregate rating']]
dfCleanIndia.head()


# CREATING NEW CSV FILE

dfCleanIndia.to_csv(r'D:\ZomatoIndiaCleaned1.csv', index=None, header=True)


#