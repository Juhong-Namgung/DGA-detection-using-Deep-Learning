import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv("./dga_label.csv")

# 1) data distribution analysis

# Count number of the data in class
print(df['class'].value_counts())

# Visualize number of the data in class(클래스별 데이터 수 그래프 시각화)
ax = sns.countplot(x="class", data=df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

total = float(len(df))
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.0f}'.format(height),
            ha="center")

plt.title("The number of data by class")
plt.show()

# 2) domain length analysis

# Calculate domain length
domain_length = []
for domain in df['domain']:
    domain_length.append(len(domain))
df['length'] = domain_length

# Count domain length
domain_count = []
for i in range(min(domain_length), max(domain_length)+1):
    domain_count.append(domain_length.count(i))
print('min len= ' + str(min(domain_length)))
print('max len= ' + str(max(domain_length)))
print(domain_count)

# Visualization
ax = sns.countplot(x='length', data=df)

total = float(len(df))
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.0f}'.format(height),
            ha="center")

plt.title("domain length")
plt.show()