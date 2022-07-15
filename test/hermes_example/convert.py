import pandas as pd
df = pd.read_csv("C60-Ih.xyz",delim_whitespace=True)
print(df)
df = df.iloc[:,1:4]
print(df)
df.to_csv("out.xyz",index=False,sep=' ')
