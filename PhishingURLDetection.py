#Importing necessary libraries
# pip install tldextract
import pandas as pd
import numpy as np
import seaborn as sns
import math
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
from scipy.stats import randint, uniform
import random
from sklearn.model_selection import KFold, cross_val_score
import re
import tldextract
from urllib.parse import urlparse
from collections import Counter
from scipy.stats import entropy
warnings.filterwarnings("ignore")

#Import raw dataset
X = pd.read_csv('phishing_site_urls.csv')

X['URL'].str.strip()  # Remove unnecessary whitespaces


# Feature Set (from original work)

# 1.  URL Length
# 2.  Number of dots
# 3.  Number of slashes
# 4.  Percentage of numerical characters in URL
# 5.  Dangerous characters
# 6.  Dangerous top-level domains
# 7.  Entropy of URL
# 8.  IP address
# 9.  Domain name length
# 10. Suspicious keywords
# 11. Repetitions
# 12. Redirections

# Modifications to original feature set (Proposal)

# Group 2,3,4,5 into "Percentage of non-alphabetical characters" or "Percentage of non-alphanumerical characters"
# Remove 11
# Research more on 10
# Add "URL Shortening Service"
# Add presence of DNS records (using WHOIS to verify)

# Feature Extraction

#1 URL length
X['URL length'] = X['URL'].apply(len)

#2 Percentage of non-alphabetical characters
X['Percentage of alphabetical characters'] = X['URL'].apply(lambda x: sum(c.isalpha() for c in x))/X['URL length']

#7 Entropy
def urlentropy(url):
    frequencies = Counter(url)
    prob = [frequencies[char] / len(url) for char in url]
    return entropy(prob, base=2)
X['Entropy'] = X['URL'].apply(urlentropy)

#8 IP Address
ip_pattern = r'[0-9]+(?:\.[0-9]+){3}'
X['IP Address'] = X['URL'].apply(lambda x: bool(re.search(ip_pattern, x)))

#9 Domain name length
X['Domain name length'] = X['URL'].apply(lambda x: len(tldextract.extract(x).domain))

#10 Suspicious keywords
sus_words = ['secure', 'account', 'update', 'login', 'verify' ,'signin', 'bank',
            'notify', 'click', 'inconvenient']

X['Suspicious keywords'] = X['URL'].apply(lambda x: sum([word in x for word in sus_words]) != 0)

#12 Redirections
def redirection(url):
    pos = url.rfind('//') #If the // is not found, it returns -1
    return pos>7
X['Redirections'] = X['URL'].apply(redirection)

#13 URL Shortening Services
shortening_services = r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|" \
                      r"yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|" \
                      r"short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|" \
                      r"doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|db\.tt|" \
                      r"qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|q\.gs|is\.gd|" \
                      r"po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|x\.co|" \
                      r"prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|" \
                      r"tr\.im|link\.zip\.net"

X["URL Shortening"] = X["URL"].apply(lambda x: 1 if re.search(shortening_services, x) else 0)

#14 DNS Records




# Feature Engineering

# Standardise numerical features
scaler = StandardScaler()
num_columns = ['URL length', 'Domain name length', 'Entropy']
X[num_columns] = scaler.fit_transform(X[num_columns])

# Cast boolean values into integers
X['IP Address'] = X['IP Address'].astype(int)
X['Suspicious keywords'] = X['Suspicious keywords'].astype(int)
X['Redirections'] = X['Redirections'].astype(int)
X['URL Shortening'] = X['URL Shortening'].astype(int)
X['Label'] = (X['Label'] == 'good').astype(int)

X.drop(columns=['URL'], inplace=True)


# Correlation Matrix
corr_matrix = X.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, annot_kws={"size": 6})
plt.show()
sns.heatmap(corr_matrix[['Label']].sort_values(by='Label').T, annot=True, cmap='coolwarm', linewidths=0.5, annot_kws={"size": 8})
plt.show()
print(corr_matrix[['Label']].sort_values(by='Label'))

# Apply PCA on Entropy and URL length because they are highly correlated
pca = PCA(n_components=1)
X['Entropy and length (PCA)'] = pca.fit_transform(X[['Entropy', 'URL length']])
X.drop(columns=['Entropy', 'URL length'], inplace=True)


# Data Splitting
X['Label'].value_counts(normalize=True)

n_samples = X['Label'].value_counts()[0]
X_good = X[X['Label'] == 1]
X_bad = X[X['Label'] == 0]
X_goodsample = X_good.sample(n=n_samples, random_state=22)
X_goodmissing = X_good.drop(X_goodsample.index)

X = pd.concat([X_bad, X_goodsample], ignore_index=True)

y = X['Label']
X.drop(columns=['Label'], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)

y_goodmissing = X_goodmissing['Label']
X_goodmissing.drop(columns=['Label'], inplace=True)

# Merging X_test and X_goodmissing
X_test = pd.concat([X_test, X_goodmissing], axis=0)

# Merging y_test and y_goodmissing
y_test = pd.concat([y_test, y_goodmissing], axis=0)


# ML Models
# XGBoost and Random Forest
kf = KFold(n_splits=3, shuffle=True, random_state=22)

xgb_model = XGBClassifier(random_state=22)
#print(cross_val_score(xgb_model, X_train, y_train, cv=kf, scoring='accuracy').mean())

rf_model = RandomForestClassifier(random_state=22)
#print(cross_val_score(rf_model, X_train, y_train, cv=kf, scoring='accuracy').mean())

rf_model.fit(X_train, y_train)
importances = rf_model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

plt.title('Feature Importance (RandomForestClassifier)')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.show()

xgb_model.fit(X_train, y_train)
importances = xgb_model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

plt.title('Feature Importance (XGBClassifier)')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.show()



rf_pred = rf_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)

# Original Score: 86.4%
print(accuracy_score(y_test, rf_pred))
print(accuracy_score(y_test, xgb_pred))
