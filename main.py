import pandas as pd
import matplotlib as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
import seaborn as sns
from collections import Counter

bank_features = pd.read_csv('/Users/etienne/PycharmProjects/InternTest/bank_transactions_dataset/bank_transaction_features.csv')
bank_labels = pd.read_csv('/Users/etienne/PycharmProjects/InternTest/bank_transactions_dataset/bank_transaction_labels.csv')

print(bank_features['bank_transaction_amount'].describe())

bank_features['bank_transaction_amount'].plot.box()

plt.pyplot.show()

bank_features['bank_transaction_type'].value_counts().plot.pie()

plt.pyplot.show()

bytype = bank_features.loc[bank_features['bank_transaction_type'] == 'DEB']

bytype['bank_transaction_amount'].plot.box()

plt.pyplot.show()

bank_labels['bank_transaction_category'].value_counts().plot.pie()

plt.pyplot.show()

bank = bank_features.merge(bank_labels)
bank = bank.drop(columns = ['bank_transaction_id', 'bank_transaction_type', 'bank_transaction_dataset', 'bank_transaction_description'])
sns.boxplot(x = 'bank_transaction_category', y = 'bank_transaction_amount',data=bank)
plt.pyplot.show()

bank = bank_features.merge(bank_labels)
bank = bank.drop(columns = ['bank_transaction_id', 'bank_transaction_amount', 'bank_transaction_dataset', 'bank_transaction_description'])
print(bank['bank_transaction_category'].unique())
for cat in bank['bank_transaction_category'].unique():
    bank_type = bank.loc[bank['bank_transaction_category'] == f'{cat}']
    bank_counts = bank_type['bank_transaction_type'].value_counts()
    bank_counts.plot.pie()
    plt.pyplot.show()


# isolate data for training and validation
train = bank_labels.loc[bank_labels['bank_transaction_dataset'] == 'TRAIN']
X_train_id = train['bank_transaction_id']
X_train = bank_features.loc[bank_features['bank_transaction_id'].isin(X_train_id)]

val = bank_labels.loc[bank_labels['bank_transaction_dataset'] == 'VAL']
X_val_id = val['bank_transaction_id']
X_val = bank_features.loc[bank_features['bank_transaction_id'].isin(X_val_id)]

# comparing training and validation data
train['bank_transaction_category'].value_counts().plot.pie()
plt.pyplot.show()
val['bank_transaction_category'].value_counts().plot.pie()
plt.pyplot.show()


# splitting train into train and dev sets
# first shuffle training data to ensure unbiased split for train and dev
X_train = X_train.sample(frac=1).reset_index(drop=True)
train1, train2, train3, train4, dev_X = np.array_split(X_train, 5)
train_X = train1.append((train2, train3, train4))

# sort labels into train and dev
train_labels = train.loc[train['bank_transaction_id'].isin(train_X['bank_transaction_id'])]
dev_labels = train.loc[train['bank_transaction_id'].isin(dev_X['bank_transaction_id'])]

# sort labels and descriptions to match
train_X = train_X.sort_values('bank_transaction_id')
train_labels = train_labels.sort_values('bank_transaction_id')

dev_X = dev_X.sort_values('bank_transaction_id')
dev_labels = dev_labels.sort_values('bank_transaction_id')

# feature representation
count_vec = CountVectorizer()
X_train = count_vec.fit_transform(train_X['bank_transaction_description'].values.astype('U'))
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train)

X_dev = count_vec.transform(dev_X['bank_transaction_description'].values.astype('U'))
X_dev_tfidf = tfidf_transformer.transform(X_dev)

# evaluate model
clf_desc = MultinomialNB().fit(X_train_tfidf, train_labels['bank_transaction_category'].values.astype('U'))
predicted = clf_desc.predict(X_dev_tfidf)
accuracy = np.mean(predicted == dev_labels['bank_transaction_category'].values.astype('U'))
print(accuracy, 'Multinomial Naive Bayes')

# use more features
X_amount = train_X['bank_transaction_amount'].values.astype('U')

dev_amount = dev_X['bank_transaction_amount'].values.astype('U')

val_amount = X_val['bank_transaction_amount'].values.astype('U')


# evaluate model
svm_model_linear = SVC(kernel='linear', C=1).fit(X_train_tfidf, train_labels['bank_transaction_category'].values.astype('U'))
predicted = svm_model_linear.predict(X_dev_tfidf)
accuracy = np.mean(predicted == dev_labels['bank_transaction_category'].values.astype('U'))
print(accuracy, 'Support Vector Machine')


# model for bank amount
clf_amount = GaussianNB()
clf_amount.fit(X_amount.reshape(-1,1), train_labels['bank_transaction_category'].values.astype('U'))
pred = clf_amount.predict(dev_amount.reshape(-1,1))
accuracy = np.mean(pred==dev_labels['bank_transaction_category'].values.astype('U'))
print(accuracy, 'amounts')

# doesnt take into account correlations between features

# model for transaction type
encoder = OrdinalEncoder()
train_cat = encoder.fit_transform((train_X['bank_transaction_type'].values.astype('U')).reshape(-1,1))
dev_cat = encoder.fit_transform((dev_X['bank_transaction_type'].values.astype('U')).reshape(-1,1))
clf_type = CategoricalNB()
clf_type.fit(train_cat, train_labels['bank_transaction_category'].values.astype('U'))
predicted = clf_type.predict(dev_cat)
accuracy = np.mean(predicted == dev_labels['bank_transaction_category'].values.astype('U'))
print(accuracy, 'transaction type')

# combine features
# weighted probabilites
total_probs = 0.91*clf_desc.predict_proba(X_dev_tfidf) + 0.6*clf_amount.predict_proba(dev_amount.reshape(-1,1)) + 0.6*clf_type.predict_proba(dev_cat)
index = clf_desc.classes_

predicted = []
for probs in total_probs:
    max_index = np.nanargmax(probs)
    predicted.append(index[max_index])

accuracy = np.mean(np.array(predicted) == dev_labels['bank_transaction_category'].values.astype('U'))
print(accuracy, 'NB combination')


# test data on the selection of models

#svm
val_X = count_vec.transform(X_val['bank_transaction_description'].values.astype('U'))
X_val_tfidf = tfidf_transformer.transform(val_X)
predicted = svm_model_linear.predict(X_val_tfidf)
accuracy = np.mean(predicted == val['bank_transaction_category'].values.astype('U'))

errors = []
for pred, label in zip(predicted, val['bank_transaction_category'].values.astype('U')) :
    if pred != label:
        errors.append((pred, label))
print(Counter(errors))
print('Test',accuracy, 'Support Vector Machine')

#Multinomial NB
predicted = clf_desc.predict(X_val_tfidf)
accuracy = np.mean(predicted == val['bank_transaction_category'].values.astype('U'))
print('Test',accuracy, 'Multinomial Naive Bayes')

#Gaussian NB
pred = clf_amount.predict(val_amount.reshape(-1,1))
accuracy = np.mean(pred==val['bank_transaction_category'].values.astype('U'))
print('Test',accuracy, 'Gaussian NB')

#Categorical NB
val_cat = encoder.fit_transform((X_val['bank_transaction_type'].values.astype('U')).reshape(-1,1))
predicted = clf_type.predict(val_cat)
accuracy = np.mean(predicted == val['bank_transaction_category'].values.astype('U'))
print('Test', accuracy, 'Categorical NB')

# combine features
# weighted probabilites
total_probs = 0.91*clf_desc.predict_proba(X_val_tfidf) + 0.6*clf_amount.predict_proba(val_amount.reshape(-1,1)) + 0.6*clf_type.predict_proba(val_cat)
index = clf_desc.classes_

predicted = []
for probs in total_probs:
    max_index = np.nanargmax(probs)
    predicted.append(index[max_index])

accuracy = np.mean(np.array(predicted) == val['bank_transaction_category'].values.astype('U'))
errors = []
for pred, label in zip(predicted, val['bank_transaction_category'].values.astype('U')) :
    if pred != label:
        errors.append((pred, label))
print(Counter(errors))
print('Test', accuracy, 'NB combination')



