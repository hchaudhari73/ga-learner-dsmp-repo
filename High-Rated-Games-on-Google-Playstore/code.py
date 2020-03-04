# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Code starts here
data = pd.read_csv(path)
data.Rating.plot(kind = "hist")
data = data[data["Rating"]<=5]
data.Rating.plot(kind = "hist")
plt.show()


#Code ends here


# --------------
# code starts here
total_null = data.isnull().sum()
percent_null = total_null/data.isnull().count()
missing_data = pd.concat([total_null, percent_null], axis = 1, keys = ["Total", "Percent"])
print(missing_data)

data.dropna(inplace = True)
total_null_1 = data.isnull().sum()
percent_null_1 = total_null_1/data.isnull().count()
missing_data_1 = pd.concat([total_null_1, percent_null_1], axis = 1, keys = ["Total", "Percent"])
# code ends 


# --------------

#Code starts here
sns.catplot(x = "Category", y = "Rating", data = data, kind = "box", height = 10)
plt.xticks(rotation = 90)
plt.title("Rating vs Category [BoxPlot]")
plt.show()
#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here
print(data.Installs.value_counts())
data.Installs = data.Installs.map(lambda x: int(x.replace(",","").replace("+","")))

le = LabelEncoder()


data['Installs'] = data['Installs'].astype('category')
data['Installs'] = data['Installs'].cat.codes
data["Installs"] = le.fit_transform(data[['Installs']])


sns.regplot(x = "Installs", y = "Rating", data = data)
plt.title("Rating vs Installs [RegPlot]")
#Code ends here



# --------------
#Code starts here
print(data.Price.value_counts())

data.Price = data.Price.map(lambda x: float(x.replace("$","")))
sns.regplot(x = "Price", y = "Rating", data = data)
plt.title("Rating vs Price [RegPlot]")
plt.show()
#Code ends here


# --------------
#Code starts here



data.Genres = data.Genres.map(lambda x: x.split(";")[0])

gr_mean = data.groupby(["Genres"], as_index = False)[["Rating"]].mean()

print(gr_mean.describe())

gr_mean = gr_mean.sort_values("Rating")

print(gr_mean)


# --------------

#Code starts here

#Converting the column into datetime format
data['Last Updated'] = pd.to_datetime(data['Last Updated'])

#Creating new column having `Last Updated` in days
data['Last Updated Days'] = (data['Last Updated'].max()-data['Last Updated'] ).dt.days 

#Setting the size of the figure
plt.figure(figsize = (10,10))

#Plotting a regression plot between `Rating` and `Last Updated Days`
sns.regplot(x="Last Updated Days", y="Rating", color = 'lightpink',data=data )

#Setting the title of the plot
plt.title('Rating vs Last Updated [RegPlot]',size = 20)

#Code ends here


