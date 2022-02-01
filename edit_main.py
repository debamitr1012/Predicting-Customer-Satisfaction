import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,accuracy_score, f1_score, \
    jaccard_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, \
    AdaBoostClassifier, BaggingClassifier

# prepare dataset
customer = pd.read_csv('datasets/olist_customers_dataset.csv')
order_items = pd.read_csv('datasets/olist_order_items_dataset.csv')
order_payments = pd.read_csv('datasets/olist_order_payments_dataset.csv')
order_reviews = pd.read_csv('datasets/olist_order_reviews_dataset.csv')
orders = pd.read_csv('datasets/olist_orders_dataset.csv')
products = pd.read_csv('datasets/olist_products_dataset.csv')
sellers = pd.read_csv('datasets/olist_sellers_dataset.csv')
product_translation = pd.read_csv('datasets/product_category_name_translation.csv')

A = pd.merge(orders, order_reviews, on='order_id')
A = pd.merge(A, order_payments, on='order_id')
A = pd.merge(A, customer, on='customer_id')

B = pd.merge(order_items, products, on='product_id')
B = pd.merge(B, sellers, on='seller_id')
B = pd.merge(B, product_translation, on='product_category_name')

df_ecommerce = pd.merge(A, B, on='order_id')

df_ecommerce = df_ecommerce[
    ['order_status', 'order_purchase_timestamp', 'order_delivered_customer_date', 'order_estimated_delivery_date',
     'shipping_limit_date', 'payment_sequential', 'payment_type', 'payment_installments', 'payment_value',
     'price', 'freight_value', 'product_category_name_english', 'product_name_lenght', 'product_description_lenght',
     'product_photos_qty', 'review_score']]
df_ecommerce = df_ecommerce.rename(
    columns={'product_name_lenght': 'product_name_length', 'product_description_lenght': 'product_description_length',
             'product_category_name_english': 'product_category'})

# data cleaning and preprocessing

df_ecommerce.isnull().sum()
prev_size = df_ecommerce.shape[0]
df_ecommerce.dropna(how='any', inplace=True)
current_size = df_ecommerce.shape[0]
df_ecommerce.isnull().values.any()
df_ecommerce['order_purchase_timestamp'] = pd.to_datetime(df_ecommerce['order_purchase_timestamp']).dt.date
df_ecommerce['order_estimated_delivery_date'] = pd.to_datetime(df_ecommerce['order_estimated_delivery_date']).dt.date
df_ecommerce['order_delivered_customer_date'] = pd.to_datetime(df_ecommerce['order_delivered_customer_date']).dt.date
df_ecommerce['shipping_limit_date'] = pd.to_datetime(df_ecommerce['shipping_limit_date']).dt.date
df_ecommerce['delivery_days'] = df_ecommerce['order_delivered_customer_date'].sub(
    df_ecommerce['order_purchase_timestamp'], axis=0).astype(str)
df_ecommerce['estimated_days'] = df_ecommerce['order_estimated_delivery_date'].sub(
    df_ecommerce['order_purchase_timestamp'], axis=0).astype(str)
df_ecommerce['shipping_days'] = df_ecommerce['shipping_limit_date'].sub(df_ecommerce['order_purchase_timestamp'],
                                                                        axis=0).astype(str)
df_ecommerce['delivery_days'] = df_ecommerce['delivery_days'].str.replace(" days", "").astype(int)
df_ecommerce['estimated_days'] = df_ecommerce['estimated_days'].str.replace(" days", "").astype(int)
df_ecommerce['shipping_days'] = df_ecommerce['shipping_days'].str.replace(" days", "").astype(int)
df_ecommerce.drop(['order_purchase_timestamp', 'order_delivered_customer_date', 'order_estimated_delivery_date',
                   'shipping_limit_date'], axis=1, inplace=True)

# Exploratory Data Analysis

fig = plt.figure(figsize=(20, 8))
ax = plt.axes()
sns.barplot(x=df_ecommerce.product_category.value_counts().index[:10],
            y=df_ecommerce.product_category.value_counts()[:10], ax=ax)
sns.set(font_scale=1)
ax.set_xlabel('Product category', fontsize=16)
ax.set_ylabel('The quantity of order', fontsize=16)
fig.suptitle("Top 10 best purchased product by customers", fontsize=25)
plt.show()

fig = plt.figure(figsize=(15, 8))
ax = plt.axes()
sns.barplot(x="payment_type", y="payment_value",
            data=df_ecommerce,
            ax=ax)
sns.set(font_scale=1.75)
ax.set_xlabel('Payment type', fontsize=20)
ax.set_ylabel('Payment value', fontsize=20)
fig.suptitle("Payment value by customer based on the payment type", fontsize=25)
plt.show()

fig = plt.figure(figsize=(15, 8))
ax = plt.axes()
sns.barplot(x="review_score", y="payment_value",
            data=df_ecommerce,
            ax=ax)
sns.set(font_scale=1.75)
ax.set_xlabel('Review score', fontsize=20)
ax.set_ylabel('Payment value', fontsize=20)
fig.suptitle("Customer review based on payment value", fontsize=25)
plt.show()

fig = plt.figure(figsize=(15, 8))
ax = plt.axes()
sns.barplot(x="review_score", y="freight_value",
            data=df_ecommerce,
            ax=ax)
sns.set(font_scale=1.75)
ax.set_xlabel('Review score', fontsize=20)
ax.set_ylabel('Freight value', fontsize=20)
fig.suptitle("Customer review based on freight value", fontsize=25)
plt.show()

fig = plt.figure(figsize=(15, 8))
ax = plt.axes()
sns.barplot(x="review_score", y="price",
            data=df_ecommerce,
            ax=ax)
sns.set(font_scale=1.75)
ax.set_xlabel('Review score', fontsize=20)
ax.set_ylabel('Price', fontsize=20)
fig.suptitle("Customer review based on Price", fontsize=25)
plt.show()

fig = plt.figure(figsize=(15, 8))
ax = plt.axes()
sns.scatterplot(x="payment_value", y="price", hue="review_score", sizes=(40, 400),
                palette=["green", "orange", "blue", "red", "brown"],
                data=df_ecommerce, ax=ax)
ax.set_xlabel('Payment value', fontsize=20)
ax.set_ylabel('Price', fontsize=20)
fig.suptitle('Correlation between payment value and price', fontsize=25)
plt.show()

fig = plt.figure(figsize=(15, 8))
ax = plt.axes()
sns.scatterplot(x="delivery_days", y="estimated_days",
                hue="review_score", sizes=(40, 400),
                palette=["green", "orange", "blue", "red", "brown"],
                data=df_ecommerce, ax=ax)
ax.set_xlabel('Delivery days', fontsize=20)
ax.set_ylabel('Estimated days', fontsize=20)
fig.suptitle('Correlation between delivery days and estimated days', fontsize=25)
plt.show()

# Feature Engineering

df_ecommerce['arrival_time'] = (df_ecommerce['estimated_days'] - df_ecommerce['delivery_days'])
delivery_arrival = []
d_arrival = df_ecommerce.arrival_time.values.tolist()
for i in d_arrival:
    if i <= 0:
        delivery_arrival.append('Late')
    else:
        delivery_arrival.append('On time')
df_ecommerce['delivery_arrival'] = delivery_arrival
df_ecommerce.loc[df_ecommerce['review_score'] < 3, 'Score'] = 0
df_ecommerce.loc[df_ecommerce['review_score'] > 3, 'Score'] = 1
df_ecommerce.drop(df_ecommerce[df_ecommerce['review_score'] == 3].index, inplace=True)
df_ecommerce.drop('review_score', axis=1, inplace=True)

df_ecommerce['order_status'] = df_ecommerce['order_status'].replace(['canceled', 'delivered'], [0, 1])
df_ecommerce['delivery_arrival'] = df_ecommerce['delivery_arrival'].replace(['Late', 'On time'], [0, 1])
one_hot_payment_type = pd.get_dummies(df_ecommerce['payment_type'])
df_ecommerce = df_ecommerce.join(one_hot_payment_type)
top_10_product_category = [x for x in
                           df_ecommerce.product_category.value_counts().sort_values(ascending=False).head(10).index]
for label in top_10_product_category:
    df_ecommerce[label] = np.where(df_ecommerce['product_category'] == label, 1, 0)

df_ecommerce.drop(['payment_type', 'product_category'], axis=1, inplace=True)

# Machine Learning Modeling

X = df_ecommerce.drop(columns='Score').to_numpy()
y = df_ecommerce[['Score']].to_numpy()
y = y.reshape(len(y), )
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# decision tree classifier

dt_clf = DecisionTreeClassifier(random_state=42)
param = {'max_depth': [1, 2, 3, 4, 5], 'min_samples_split': [5, 10, 100, 300, 500, 1000]}
dt_clf_gridcv = GridSearchCV(dt_clf, param, cv=3, refit=True, return_train_score=True, scoring='accuracy')
dt_clf_gridcv.fit(X_train, y_train)
cv_result = pd.DataFrame(dt_clf_gridcv.cv_results_)
retain_cols = ['params', 'mean_test_score', 'rank_test_score']
cv_result[retain_cols].sort_values('rank_test_score')

# plotting decision tree confusion matrix

fig, ax = plt.subplots(figsize=(10, 7))
sns.set(font_scale=1.75)
y_test_pred = dt_clf_gridcv.best_estimator_.predict(X_test)
y_train_pred = dt_clf_gridcv.best_estimator_.predict(X_train)
cm = confusion_matrix(y_test, y_test_pred, labels=dt_clf_gridcv.best_estimator_.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=dt_clf_gridcv.best_estimator_.classes_)
disp.plot(ax=ax)
fig.suptitle('Decision tree confusion matrix', fontsize=25)
plt.show()

# accuracy calc
accuracy_training = accuracy_score(y_train, y_train_pred)
accuracy_test = accuracy_score(y_test, y_test_pred)
f1_test = f1_score(y_test, y_test_pred)
f1_train = f1_score(y_train, y_train_pred)
jaccard_test = jaccard_score(y_test, y_test_pred)
jaccard_train = jaccard_score(y_train, y_train_pred)
print(f'Decision tree model')
print(f'Accuracy Training Data: {accuracy_training}')
print(f'Accuracy Test Data: {accuracy_test}')
print(f'f1 Training Data: {f1_train}')
print(f'f1 Test Data: {f1_test}')
print(f'Jaccard Training Data: {jaccard_train}')
print(f'Jaccard Test Data: {jaccard_test}')
'''
Decision tree model
Accuracy Training Data: 0.8714251265975403
Accuracy Test Data: 0.8672775500361707
f1 Training Data: 0.9282127229888926
f1 Test Data: 0.9259498439349908
Jaccard Training Data: 0.8660419309859686
Jaccard Test Data: 0.8621104319070048
'''

# random forest classifier

rf_clf = RandomForestClassifier(random_state=42)
parameters = {
    'n_estimators': (10, 20, 30, 40, 50),
    'max_depth': (1, 2, 3, 4, 5)
}
rf_clf_gridcv = GridSearchCV(rf_clf, parameters, cv=5,
                             scoring='accuracy')
rf_clf_gridcv.fit(X_train, y_train)

GridSearchCV(cv=5, estimator=RandomForestClassifier(random_state=42),
             param_grid={'max_depth': (1, 2, 3, 4, 5),
                         'n_estimators': (10, 20, 30, 40, 50)},
             scoring='accuracy')
cv_result = pd.DataFrame(rf_clf_gridcv.cv_results_)
retain_cols = ['params', 'mean_test_score', 'rank_test_score']
cv_result[retain_cols].sort_values('rank_test_score')

# random forest confusion matrix

fig, ax = plt.subplots(figsize=(10, 7))
y_test_pred = rf_clf_gridcv.best_estimator_.predict(X_test)
y_train_pred = rf_clf_gridcv.best_estimator_.predict(X_train)
cm = confusion_matrix(y_test, y_test_pred, labels=rf_clf_gridcv.best_estimator_.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=rf_clf_gridcv.best_estimator_.classes_)
disp.plot(ax=ax)
fig.suptitle('Random forest confusion matrix', fontsize=25)
plt.show()

# accuracy calc
accuracy_training = accuracy_score(y_train, y_train_pred)
accuracy_test = accuracy_score(y_test, y_test_pred)
f1_test = f1_score(y_test, y_test_pred)
f1_train = f1_score(y_train, y_train_pred)
jaccard_test = jaccard_score(y_test, y_test_pred)
jaccard_train = jaccard_score(y_train, y_train_pred)
print(f'Random Forest model')
print(f'Accuracy Training Data: {accuracy_training}')
print(f'Accuracy Test Data: {accuracy_test}')
print(f'f1 Training Data: {f1_train}')
print(f'f1 Test Data: {f1_test}')
print(f'Jaccard Training Data: {jaccard_train}')
print(f'Jaccard Test Data: {jaccard_test}')
'''
Random forest model
Accuracy Training Data: 0.870122980467808
Accuracy Test Data: 0.8664094526163492
f1 Training Data: 0.927430003503193
f1 Test Data: 0.925397252895233
Jaccard Training Data: 0.864680166825787
Jaccard Test Data: 0.861152882205513
'''

# gradient boosting classifier

gb_clf = GradientBoostingClassifier(random_state=42)
parameters = {
    'n_estimators': (10, 20, 30, 40, 50),
    'max_depth': (1, 2, 3, 4, 5)
}
gb_clf_gridcv = GridSearchCV(gb_clf, parameters, cv=5,
                             scoring='accuracy')
gb_clf_gridcv.fit(X_train, y_train)

GridSearchCV(cv=5, estimator=GradientBoostingClassifier(random_state=42),
             param_grid={'max_depth': (1, 2, 3, 4, 5),
                         'n_estimators': (10, 20, 30, 40, 50)},
             scoring='accuracy')
cv_result = pd.DataFrame(gb_clf_gridcv.cv_results_)
retain_cols = ['params', 'mean_test_score', 'rank_test_score']
cv_result[retain_cols].sort_values('rank_test_score')

# gradient boosting confusion matrix

fig, ax = plt.subplots(figsize=(10, 7))
y_test_pred = gb_clf_gridcv.best_estimator_.predict(X_test)
y_train_pred = gb_clf_gridcv.best_estimator_.predict(X_train)
cm = confusion_matrix(y_test, y_test_pred, labels=gb_clf_gridcv.best_estimator_.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=gb_clf_gridcv.best_estimator_.classes_)
disp.plot(ax=ax)
fig.suptitle('Gradient boosting confusion matrix', fontsize=25)
plt.show()

# accuracy calc
accuracy_training = accuracy_score(y_train, y_train_pred)
accuracy_test = accuracy_score(y_test, y_test_pred)
f1_test = f1_score(y_test, y_test_pred)
f1_train = f1_score(y_train, y_train_pred)
jaccard_test = jaccard_score(y_test, y_test_pred)
jaccard_train = jaccard_score(y_train, y_train_pred)
print(f'Gradient Boosting model')
print(f'Accuracy Training Data: {accuracy_training}')
print(f'Accuracy Test Data: {accuracy_test}')
print(f'f1 Training Data: {f1_train}')
print(f'f1 Test Data: {f1_test}')
print(f'Jaccard Training Data: {jaccard_train}')
print(f'Jaccard Test Data: {jaccard_test}')

'''
Gradient Boosting Model
Accuracy Training Data: 0.877525922353508
Accuracy Test Data: 0.87089462261876
f1 Training Data: 0.931319387161769
f1 Test Data: 0.927701407081319
Jaccard Training Data: 0.871466531696824
Jaccard Test Data: 0.865152125730405
'''

# extra trees classifier

ext_clf = ExtraTreesClassifier(random_state=42)
parameters = {
    'n_estimators': (10, 20, 30, 40, 50),
    'max_depth': (1, 2, 3, 4, 5)
}
ext_clf_gridcv = GridSearchCV(ext_clf, parameters, cv=5,
                              scoring='accuracy')
ext_clf_gridcv.fit(X_train, y_train)

GridSearchCV(cv=5, estimator=ExtraTreesClassifier(random_state=42),
             param_grid={'max_depth': (1, 2, 3, 4, 5),
                         'n_estimators': (10, 20, 30, 40, 50)},
             scoring='accuracy')
cv_result = pd.DataFrame(ext_clf_gridcv.cv_results_)
retain_cols = ['params', 'mean_test_score', 'rank_test_score']
cv_result[retain_cols].sort_values('rank_test_score')

# extra trees confusion matrix

fig, ax = plt.subplots(figsize=(10, 7))
y_test_pred = ext_clf_gridcv.best_estimator_.predict(X_test)
y_train_pred = ext_clf_gridcv.best_estimator_.predict(X_train)
cm = confusion_matrix(y_test, y_test_pred, labels=ext_clf_gridcv.best_estimator_.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=ext_clf_gridcv.best_estimator_.classes_)
disp.plot(ax=ax)
fig.suptitle('Extra Trees confusion matrix', fontsize=25)
plt.show()

# accuracy calc
accuracy_training = accuracy_score(y_train, y_train_pred)
accuracy_test = accuracy_score(y_test, y_test_pred)
f1_test = f1_score(y_test, y_test_pred)
f1_train = f1_score(y_train, y_train_pred)
jaccard_test = jaccard_score(y_test, y_test_pred)
jaccard_train = jaccard_score(y_train, y_train_pred)
print(f'Extra Trees model')
print(f'Accuracy Training Data: {accuracy_training}')
print(f'Accuracy Test Data: {accuracy_test}')
print(f'f1 Training Data: {f1_train}')
print(f'f1 Test Data: {f1_test}')
print(f'Jaccard Training Data: {jaccard_train}')
print(f'Jaccard Test Data: {jaccard_test}')

'''
Extra Trees model
Accuracy Training Data: 0.851977333011815
Accuracy Test Data: 0.851073064866168
f1 Training Data: 0.918715281685944
f1 Test Data: 0.91816832732669
Jaccard Training Data: 0.849651591412861
Jaccard Test Data: 0.848716441309033
'''

# Adaptive Boost classifier with Random Forest Classifier as base estimator

base = RandomForestClassifier(max_depth=5)
ada_clf = AdaBoostClassifier(random_state=42, base_estimator=base)
parameters = {
    'n_estimators': (10, 20, 30, 40, 50),
}
ada_clf_gridcv = GridSearchCV(ada_clf, parameters, cv=5,
                              scoring='accuracy')
ada_clf_gridcv.fit(X_train, y_train)

GridSearchCV(cv=5, estimator=AdaBoostClassifier(random_state=42),
             param_grid={'max_depth': (1, 2, 3, 4, 5),
                         'n_estimators': (10, 20, 30, 40, 50)},
             scoring='accuracy')
cv_result = pd.DataFrame(ada_clf_gridcv.cv_results_)
retain_cols = ['params', 'mean_test_score', 'rank_test_score']
cv_result[retain_cols].sort_values('rank_test_score')

# Adaptive boost confusion matrix

fig, ax = plt.subplots(figsize=(10, 7))
y_test_pred = ada_clf_gridcv.best_estimator_.predict(X_test)
y_train_pred = ada_clf_gridcv.best_estimator_.predict(X_train)
cm = confusion_matrix(y_test, y_test_pred, labels=ada_clf_gridcv.best_estimator_.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=ada_clf_gridcv.best_estimator_.classes_)
disp.plot(ax=ax)
fig.suptitle('Adaptive Boost confusion matrix', fontsize=25)
plt.show()

# accuracy calc
accuracy_training = accuracy_score(y_train, y_train_pred)
accuracy_test = accuracy_score(y_test, y_test_pred)
f1_test = f1_score(y_test, y_test_pred)
f1_train = f1_score(y_train, y_train_pred)
jaccard_test = jaccard_score(y_test, y_test_pred)
jaccard_train = jaccard_score(y_train, y_train_pred)
print(f'Adaptive Boost model, Random Forest Base Estimator')
print(f'Accuracy Training Data: {accuracy_training}')
print(f'Accuracy Test Data: {accuracy_test}')
print(f'f1 Training Data: {f1_train}')
print(f'f1 Test Data: {f1_test}')
print(f'Jaccard Training Data: {jaccard_train}')
print(f'Jaccard Test Data: {jaccard_test}')
'''
Adaptive Boost model, Random Forest Base Estimator
Accuracy Training Data: 0.8878345792138895
Accuracy Test Data: 0.8772124427296841
f1 Training Data: 0.936760317320046
f1 Test Data: 0.9309727795249972
Jaccard Training Data: 0.8810434115465763
Jaccard Test Data: 0.8708597514582805'''


# Bagging Classifier

b_clf = BaggingClassifier(random_state=42)
parameters = {
    'n_estimators': (10, 20, 30, 40, 50)
}
b_clf_gridcv = GridSearchCV(b_clf, parameters, cv=5,
                            scoring='accuracy')
b_clf_gridcv.fit(X_train, y_train)

GridSearchCV(cv=5, estimator=BaggingClassifier(random_state=42),
             param_grid={'max_depth': (1, 2, 3, 4, 5),
                         'n_estimators': (10, 20, 30, 40, 50)},
             scoring='accuracy')
cv_result = pd.DataFrame(b_clf_gridcv.cv_results_)
retain_cols = ['params', 'mean_test_score', 'rank_test_score']
cv_result[retain_cols].sort_values('rank_test_score')

# bagging classifier confusion matrix

fig, ax = plt.subplots(figsize=(10, 7))
y_test_pred = b_clf_gridcv.best_estimator_.predict(X_test)
y_train_pred = b_clf_gridcv.best_estimator_.predict(X_train)
cm = confusion_matrix(y_test, y_test_pred, labels=b_clf_gridcv.best_estimator_.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=b_clf_gridcv.best_estimator_.classes_)
disp.plot(ax=ax)
fig.suptitle('Bagging Classifier confusion matrix', fontsize=25)
plt.show()

# accuracy calc
accuracy_training = accuracy_score(y_train, y_train_pred)
accuracy_test = accuracy_score(y_test, y_test_pred)
f1_test = f1_score(y_test, y_test_pred)
f1_train = f1_score(y_train, y_train_pred)
jaccard_test = jaccard_score(y_test, y_test_pred)
jaccard_train = jaccard_score(y_train, y_train_pred)
print(f'Bagging Classifier model')
print(f'Accuracy Training Data: {accuracy_training}')
print(f'Accuracy Test Data: {accuracy_test}')
print(f'f1 Training Data: {f1_train}')
print(f'f1 Test Data: {f1_test}')
print(f'Jaccard Training Data: {jaccard_train}')
print(f'Jaccard Test Data: {jaccard_test}')

'''
Bagging Classifier model 
Accuracy Training Data: 0.9987943091391367
Accuracy Test Data: 0.9075958524234387
f1 Training Data: 0.9992815782289468
f1 Test Data: 0.9469105015239679
Jaccard Training Data: 0.9985641879765101
Jaccard Test Data: 0.899173814660843
'''

# Adaptive Boost classifier with decision tree classifier as the base estimator
base = DecisionTreeClassifier(max_depth=5, min_samples_split=100)
ada_clf = AdaBoostClassifier(random_state=42, base_estimator=base)
parameters = {
    'n_estimators': (10, 20, 30, 40, 50),
}
ada_clf_gridcv = GridSearchCV(ada_clf, parameters, cv=5,
                              scoring='accuracy')
ada_clf_gridcv.fit(X_train, y_train)

GridSearchCV(cv=5, estimator=AdaBoostClassifier(random_state=42),
             param_grid={'max_depth': (1, 2, 3, 4, 5),
                         'n_estimators': (10, 20, 30, 40, 50)},
             scoring='accuracy')
cv_result = pd.DataFrame(ada_clf_gridcv.cv_results_)
retain_cols = ['params', 'mean_test_score', 'rank_test_score']
cv_result[retain_cols].sort_values('rank_test_score')

# Adaptive boost confusion matrix

fig, ax = plt.subplots(figsize=(10, 7))
y_test_pred = ada_clf_gridcv.best_estimator_.predict(X_test)
y_train_pred = ada_clf_gridcv.best_estimator_.predict(X_train)
cm = confusion_matrix(y_test, y_test_pred, labels=ada_clf_gridcv.best_estimator_.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=ada_clf_gridcv.best_estimator_.classes_)
disp.plot(ax=ax)
fig.suptitle('Adaptive Boost confusion matrix', fontsize=25)
plt.show()

# accuracy calc
accuracy_training = accuracy_score(y_train, y_train_pred)
accuracy_test = accuracy_score(y_test, y_test_pred)
f1_test = f1_score(y_test, y_test_pred)
f1_train = f1_score(y_train, y_train_pred)
jaccard_test = jaccard_score(y_test, y_test_pred)
jaccard_train = jaccard_score(y_train, y_train_pred)
print(f'Adaptive Boost model, Decision Tree Base Estimator')
print(f'Accuracy Training Data: {accuracy_training}')
print(f'Accuracy Test Data: {accuracy_test}')
print(f'f1 Training Data: {f1_train}')
print(f'f1 Test Data: {f1_test}')
print(f'Jaccard Training Data: {jaccard_train}')
print(f'Jaccard Test Data: {jaccard_test}')

'''
Adaptive Boost model, Decision Tree Base Estimator
Accuracy Training Data: 0.8833494092114782
Accuracy Test Data: 0.8706534844465879
f1 Training Data: 0.933790469933688
f1 Test Data: 0.9267373251748253
Jaccard Training Data: 0.8758039049562907
Jaccard Test Data: 0.8634767116314583'''
