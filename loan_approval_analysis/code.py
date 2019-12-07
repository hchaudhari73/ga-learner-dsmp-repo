# --------------
import numpy as np
import pandas as pd
from scipy.stats import mode 

bank = pd.read_csv(path)

categorical_var = bank.select_dtypes(include = "object")

print (categorical_var)

numerical_var = bank.select_dtypes(include = "number")

print(numerical_var)


# --------------
# code starts here
banks = bank.drop(["Loan_ID"], axis = 1)
print(banks.isnull().sum())
bank_mode = banks.mode
banks.fillna(bank_mode, inplace = True)
print(banks.isnull())

#code ends here


# --------------





avg_loan_amount = pd.pivot_table(banks, index = ["Gender","Married","Self_Employed"], values = ["LoanAmount"])







# --------------
# code starts here



loan_approved_se = 0 
for i in range(len(banks)):
    if banks["Self_Employed"][i]=="Yes" and banks["Loan_Status"][i]=="Y":
        loan_approved_se+=1

loan_approved_nse = 0 
for i in range(len(banks)):
    if banks["Self_Employed"][i]=="No" and banks["Loan_Status"][i]=="Y":
        loan_approved_nse+=1

Loan_Status = 614

percentage_se = loan_approved_se/Loan_Status*100
percentage_nse = loan_approved_nse/Loan_Status*100






# code ends here


# --------------
# code starts here
loan_term = banks["Loan_Amount_Term"].apply(lambda x:x/12)
big_loan_term = 0
for i in loan_term:
    if i >= 25:
        big_loan_term+=1
        
big_loan_term




# code ends here


# --------------
# code starts here


loan_groupby = banks.groupby("Loan_Status")
loan_groupby = loan_groupby[["ApplicantIncome","Credit_History"]]
mean_values = loan_groupby.mean()
mean_values


# code ends here


