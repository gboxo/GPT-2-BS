import pandas as pd
import matplotlib.pyplot as plt

# Read the data
data = pd.read_csv("attributions_mask.csv")

repeated_values = ["The","noun","is","adjective","but","the","neg_noun","iss"]
data['tokens'] = pd.Series(repeated_values * (len(data) // len(repeated_values)) + repeated_values[:len(data) % len(repeated_values)])
df={"tokens":data.iloc[:,1],"attributions":data.iloc[:,2]}
df=pd.DataFrame(df)
group_mean = df.groupby('tokens',sort=False).mean()
plt.figure(figsize=(10, 6))
group_mean.plot.bar() 
plt.title('Barplot for aggregated')
plt.xlabel('tokens')
plt.ylabel('attributions')
plt.xticks(rotation=45, ha="right")
plt.savefig('Barplot_aggregate_mask_but.png', bbox_inches='tight')
plt.close()
