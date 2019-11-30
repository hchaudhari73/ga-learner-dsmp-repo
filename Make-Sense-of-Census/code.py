# --------------
# Importing header files
import numpy as np

# Path of the file has been stored in variable called 'path'

#New record
new_record=[[50,  9,  4,  1,  0,  0, 40,  0]]

#Code starts here
data = np.genfromtxt(path, delimiter=',', skip_header=1)
census = np.concatenate((data, np.array(new_record)))


# --------------
#Code starts here
import numpy as np
age = np.array([census[i][0] for i in range (len(census))])
max_age = max(age)
min_age = min(age)
age_mean = np.mean(age)
age_std =  np.std(age)


# --------------
#Code starts here
race_0 = census[census[:,2]==0]
race_1 = census[census[:,2]==1]
race_2 = census[census[:,2]==2]
race_3 = census[census[:,2]==3] 
race_4 = census[census[:,2]==4]


len_0 = len(race_0)
len_1 = len(race_1)
len_2 = len(race_2)
len_3 = len(race_3)
len_4 = len(race_4)

length = [len_0,len_1,len_2,len_3,len_4]
dic = {i:length[i] for i in range(len(length))}
minimum = [mini for mini in dic.keys() if dic[mini]==min(length)]
minority_race = minimum[0]







# --------------
#Code starts here
import numpy as np
senior_citizens = np.array(census[census[:,0]>60])

print(senior_citizens)
working_hours_sum = 0
for i in range(len(census)):
    if census[i][0]>60:
        working_hours_sum += census[i][6]
    
senior_citizens_len = len(senior_citizens)
avg_working_hours = round(working_hours_sum/senior_citizens_len,2)
print(f"Average working hours = {avg_working_hours}")



# --------------
#Code starts here
import numpy as np 
high = np.array(census[census[:,1]>10])
low = np.array(census[census[:,1]<=10])

avg_pay_high = np.mean(high[:,7])
avg_pay_low = round(np.mean(low[:,7]),2)

print(avg_pay_high > avg_pay_low)


