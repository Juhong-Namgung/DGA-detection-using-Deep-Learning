import pandas as pd
from sklearn.utils import shuffle

# Read dga.txt
alexa_file = pd.read_csv('alexa-top-1m.txt', header=None)

# Process Alexa data
alexa_domains = alexa_file[0].tolist()

alexa_archive = pd.DataFrame()
alexa_archive['domain'] = alexa_domains
alexa_archive['source'] = "Alexa"
alexa_archive['class'] = 0

# Read dga.txt
dga_file = pd.read_csv('dga.txt', header=None)


def extract_source(source):
    return str(source).split()[3]

dga_archive = pd.DataFrame()
dga_archive['domain'] = dga_file[0]
dga_archive['source'] = dga_file[1].map(extract_source)

# Extract DGA classes
source_counts = dga_archive['source'].value_counts(sort=True)
print(source_counts)

dga_classes_dict = dict()

for i in range(20):
    # top 20 class
    dga_classes_dict[source_counts.index[i]] = i + 1

print(dga_classes_dict)


def is_class(source):
    return (source in dga_classes_dict)


def extract_class(source):
    return dga_classes_dict[source]

# Use only top 20 class domain
dga_archive = dga_archive[dga_archive['source'].map(is_class)]
dga_archive['class'] = dga_archive['source'].map(extract_class)

# Combine Alexa and DGA data
result = pd.concat([alexa_archive, dga_archive], sort=False)

# Shuffle the data
result = shuffle(result, random_state=33)
result.to_csv("./dga_label.csv",mode='w',index=False)