import numpy as np 
import pandas as pd 

import nltk as nl
import re
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold,cross_val_score
from sklearn.metrics import roc_curve, auc

from scipy.sparse import csr_matrix, hstack

import seaborn as sns
from matplotlib import pyplot as plt
%matplotlib inline
from wordcloud import WordCloud


# Import Dataset
path = 'C:\\Users\\kriti\\Documents\\BAPM Course Material\\2nd Sem\\Data Mining\\Project\\Project Files\\Toxic Comments\\'
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')

# Check Sample Comments
#train[(train['threat']==1)&(train['insult']==0)].head(5)
#train[(train['obscene']==1)].count()

# Data Exploration
# check missing values in numeric columns
train.describe()

# check for any 'null' comment
null_comment = train[train['comment_text'].isnull()]
len(null_comment)

# Plotting the length of comments
train['char_length'] = train['comment_text'].apply(lambda x: len(str(x)))
sns.set()
train['char_length'].hist()
plt.show()

# Generating Wordclouds for the Comments
target_cols = ['obscene','insult','toxic','severe_toxic','identity_hate','threat']
for label in target_cols:
	label_comments = train[train[label]==1]
	cloud = WordCloud(width=1440, height= 1080,max_words= 100).generate(' '.join(label_comments['comment_text'].astype(str)))
	plt.figure(figsize=(10,5))
	plt.imshow(cloud)
	plt.axis('off')
	plt.show()

# Data Preprocessing
def clean_str(string):
    
    string = re.sub(r"[^A-Za-z0-9()!\'\`%$]", " ", string) # replace single characters not present in the lists by a space.
    string = re.sub(r"\'s", " \'s", string) # separate it with the word before（add space）
    string = re.sub(r"\'ve", " have", string) 
    string = re.sub(r"\'re", " are", string)
    string = re.sub(r"\'d", " would", string)
    string = re.sub(r"\'ll", " will", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\%", " % ", string) 
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\'t"," not",string)
    string = re.sub(r"\'m", " am", string)
    
    
    # removing non ascii
    string = re.sub(r'[^\x00-\x7F]+', "", string) 
    
    return string.strip().lower()

def remove_stopwords(text):
	eng_stopwords = set(stopwords.words("english"))
	#eng_stopwords
    words = text.split(' ')
    clean_words = [word for word in words if word not in eng_stopwords] 
    clean_text = " ".join(clean_words)
    return clean_text

# Cleaning
train['text_clean'] = train.comment_text.apply(clean_str)
# Stopword Removal
train['text_clean'] = train.text_clean.apply(remove_stopwords)

# Tokenization
#train['tokens'] = train.text_clean.apply(lambda x: word_tokenize(x))

# Bigrams
#for x in ngrams(train['tokens'].iloc[42],n=2):
#    print(x)

# Part of Speech Tagging
#train['tags'] = train.tokens.apply(nl.pos_tag)
#for x in train['tags'][0]:
#    print(x)

#def get_wordnet_tag():

# Lemmatization
lem = WordNetLemmatizer()
#train['tokens_final']=train.tokens.apply(lambda tokens: [lem.lemmatize(word,'v') for word in tokens])
train['text_lem']=train.text_clean.apply(lambda text_clean: ' '.join([lem.lemmatize(word,'v') for word in text_clean.split(' ')]))


# Vectorize the Text
vect = TfidfVectorizer(max_features=5000)
#vect
X_train = train['text_lem']
X_train_dtm = vect.fit_transform(X_train)
target_cols = ['obscene','insult','toxic','severe_toxic','identity_hate','threat']

# Models
logreg = LogisticRegression(C=12.0)
nb = MultinomialNB()

models = [logreg, nb]

# Classification Approach 1: Binary Relevance
for clf in models:
	print('Model: ' + str(clf).split('(')[0])
	for label in target_cols:
		print('... Processing {}'.format(label))

		y = train[label]
		
		# Training to fit and generate probabilities - Use this to generate confusion matrices
		clf.fit(X_train_dtm, y)
		y_pred_X = clf.predict(X_train_dtm)
		print('AUC = ' + str(roc_auc_score(y,y_pred_X)))
		roc(y,y_pred_X)
        print(classification_report(y,y_pred_X))

		# Cross Validated Fit and compute accuracies
		k_fold = KFold(n_splits=10, shuffle=True, random_state=None)
		cross_score = cross_val_score(clf, X_train_dtm, y, cv=k_fold, n_jobs=1)
    	print('Average Accuracy = ' + str(np.mean(cross_score)))
    	print('Fold Accuracies = ' + str(cross_score))


# Correlation Testing for Target Labels
target_data = train[target_cols]

colormap = plt.cm.magma
plt.figure(figsize=(7,7))
plt.title('Correlation of features & targets',y=1.05,size=14)
sns.heatmap(target_data.astype(float).corr(),linewidths=0.1,vmax=1.0,square=True,cmap=colormap,
           linecolor='white',annot=True)


# Classification Approach 2: Classifier Chains
def add_feature(X, feature_to_add):
    '''
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    '''
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')

for clf in models:
	X_train_dtm = vect.fit_transform(X_train)
	print('Model: ' + str(clf).split('(')[0])
	for label in target_cols:
		print('... Processing {}'.format(label))
		print('Shape of X_dtm is now {}'.format(X_train_dtm.shape))
		y = train[label]
		
		# Training to fit and generate probabilities - Use this to generate confusion matrices
		clf.fit(X_train_dtm, y)
		y_pred_X = clf.predict(X_train_dtm)
		print('AUC = ' + str(roc_auc_score(y,y_pred_X)))
		roc(y,y_pred_X)
		print(classification_report(y,y_pred_X))

		# Cross Validated Fit and compute accuracies
		k_fold = KFold(n_splits=10, shuffle=True, random_state=None)
		cross_score = cross_val_score(clf, X_train_dtm, y, cv=k_fold, n_jobs=1)
    	print('Average Accuracy = ' + str(np.mean(cross_score)))
    	print('Fold Accuracies = ' + str(cross_score))
    	X_train_dtm = add_feature(X_train_dtm, y)


def roc(y,y_pred):
    fpr, tpr, threshold = roc_curve(y, y_pred_X)
    roc_auc = auc(fpr, tpr)

    # method I: plt
    import matplotlib.pyplot as plt
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
