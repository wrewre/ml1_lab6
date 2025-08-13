import pandas as pd
import numpy as np
import math as mp
df=pd.read_csv(r"C:\Users\prana\OneDrive\Desktop\machine_learning\lab6\DCT_mal.csv")
values=df['LABEL'].value_counts().to_dict()
total=0
length=len(values)
summation=sum(values.values())
print(length)
dictionary={}
for i in values:
    dictionary[i]=values[i]/summation
print(dictionary)

for j in dictionary:
    total+=(-(dictionary[j]*mp.log2(dictionary[j])))
print(total)