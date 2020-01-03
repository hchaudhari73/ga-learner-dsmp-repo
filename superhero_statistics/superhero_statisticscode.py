# --------------
#Header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#path of the data file- path
data = pd.read_csv(path)
#Code starts here 
data.Gender.replace("-","Agender", inplace = True)
gender_count = data.Gender.value_counts()
gender_count.plot(kind = "bar")
plt.show()


# --------------
#Code starts here
alignment = data.Alignment.value_counts()
plt.pie(alignment, labels=alignment.index)
plt.show()


# --------------
sc_df = data[["Strength","Combat"]]
sc_covariance = round(sc_df.Strength.cov(sc_df.Combat),2)
print(sc_covariance)


sc_strength = round(sc_df["Strength"].std(),2)
print(sc_strength)

sc_combat = round(sc_df["Combat"].std(),2)
print(sc_combat)

sc_pearson = round((sc_covariance/(sc_strength*sc_combat)),2)
print(sc_pearson)

ic_df = data[["Intelligence","Combat"]]
ic_covariance = round(ic_df.Intelligence.cov(ic_df.Combat),2)
print(ic_covariance)

ic_intelligence = round(ic_df["Intelligence"].std(),2)
print(ic_intelligence)

ic_combat = round(ic_df["Combat"].std(),2)
print(ic_combat)

ic_pearson = round((ic_covariance/(ic_intelligence*sc_combat)),2)
print(ic_pearson)


# --------------
#Code starts here
total_high = data.Total.quantile(0.99)

super_best = data[data.Total>total_high]

super_best_names = [super_best.Name]

print(super_best_names)


# --------------
#Code starts here

fig,(ax_1,ax_2,ax_3) = plt.subplots(1,3)
ax_1.boxplot(data.Intelligence)
ax_1.set_title("Intelligence")
ax_2.boxplot(data.Speed)
ax_2.set_title("Speed")
ax_3.boxplot(data.Power)
ax_3.set_title("Power")
plt.show()


