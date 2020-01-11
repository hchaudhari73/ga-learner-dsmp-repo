# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# code starts here
df = pd.read_csv(path)
p_a = (df.fico>700).sum()/len(df)
p_b = (df.purpose=="debt_consolidation").sum()/len(df)
df1 = df[df.purpose=="debt_consolidation"]
p_b_a = (df1.fico>700).sum()/len(df)
p_a_b = p_a*p_b_a/p_b
result = p_b_a==p_a
print(result)



# code ends here


# --------------
# code starts here
prob_lp = (df["paid.back.loan"]=="Yes").sum()/len(df)

prob_cs = (df["credit.policy"] == "Yes").sum()/len(df)

new_df = df[df["paid.back.loan"]=="Yes"]

df2 = df[df["credit.policy"]=="Yes"]

prob_pd_cs = (new_df["credit.policy"]=="Yes").sum()/len(new_df)

bayes = prob_lp*prob_pd_cs/prob_cs


# code ends here


# --------------
# code starts here
df.purpose.value_counts().plot(kind = "bar")
plt.show()

df1 = df[df["paid.back.loan"]=="No"]

df1.purpose.value_counts().plot(kind = "bar")
plt.show()
# code ends here


# --------------
inst_median = df.installment.median()

inst_mean = df.installment.mean()

df.installment.plot(kind = "hist")
plt.show()
df["log.annual.inc"].plot(kind = "hist")
plt.show()



# code ends here


