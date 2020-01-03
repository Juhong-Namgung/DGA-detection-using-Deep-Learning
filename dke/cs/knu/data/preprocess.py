import pandas as pd
from sklearn.utils import shuffle

df_dga = pd.read_csv("./dga_feed.csv")
df_alexa = pd.read_csv("./top-1m.csv")

# DGA labels
dga_labels_dict = {'alexa':0, 'banjori':1, 'tinba':2, 'Post':3, 'ramnit':4, 'necurs':5, 'murofet':6, 'qakbot':7, 'shiotob/urlzone/bebloh':8, 'simda':9,
              'pykspa':10, 'ranbyus':11, 'dyre':12, 'kraken':13, 'Cryptolocker':14, 'nymaim':15, 'locky':16, 'vawtrak':17, 'shifu':18,
              'ramdo':19, 'P2P':20 }

# Process DGA data
dga_labels= []
dga_labels_str = []
dga_domains = []
z = 0
for x in df_dga['source'].tolist():
    if x in dga_labels_dict:
        dga_labels.append(dga_labels_dict[x])
        dga_labels_str.append(df_dga['source'].tolist()[z])
        dga_domains.append(df_dga['domain'].tolist()[z])
    z = z + 1

# Data columns("domain", "source", "class")
dga_archive = pd.DataFrame(columns=['domain'])
dga_archive['domain'] = dga_domains
dga_archive['source'] = dga_labels_str
dga_archive['class'] = dga_labels

# Process Alexa data
alexa_domains = df_alexa['domain'].tolist()
alexa_labels = []
alexa_labels_str = []

for x in alexa_domains:
    alexa_labels.append(0)
    alexa_labels_str.append("Alexa")

alexa_archive = pd.DataFrame(columns=['domain'])
alexa_archive['domain'] = alexa_domains
alexa_archive['source'] = alexa_labels_str
alexa_archive['class'] = alexa_labels

# Combine DGA and Alexa data
result = pd.concat([dga_archive, alexa_archive])

# Shuffle the data
result = shuffle(result, random_state=33)
result.to_csv("./dga_label.csv",mode='w',index=False)

