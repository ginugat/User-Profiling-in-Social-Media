#XML - creation
from lxml import etree as ET

# pandas
import pandas as pd
import numpy as np
from pandas import Series,DataFrame
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing import image
from keras.utils import np_utils
from keras import backend as K
import xml.etree.cElementTree as ET
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model
import codecs
import pickle
import scipy.sparse as sparse
from scipy.sparse.linalg import svds
from sklearn import tree
from sklearn.linear_model import Lasso
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn import model_selection

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from sklearn.ensemble import BaggingClassifier

#System Argument
import sys
Inputpath = sys.argv[1]
Outputpath = sys.argv[2]

#OS to save file in particular folder
import os.path as save


# Creating data frame for profile file
train_df = pd.read_csv("/data/training/profile/profile.csv", index_col=0)
LIWC_train = pd.read_csv("/data/training/LIWC/LIWC.csv")
test_df = pd.read_csv(Inputpath+"/profile/profile.csv", index_col =0)
LIWC_test = pd.read_csv(Inputpath+"/LIWC/LIWC.csv")
relation_train_df = pd.read_csv("/data/training/relation/relation.csv",index_col=0)
relation_test_df = pd.read_csv(Inputpath+"/relation/relation.csv",index_col=0)
text_location = "/data/training/text/"
test_location = Inputpath+"/text/"
image_location_train = "/data/training/image/"
image_location_test = Inputpath+"/image/"
train_data = train_df.join(LIWC_train.set_index('userId'), on='userid')
test_data = test_df.join(LIWC_test.set_index('userId'), on ='userid')

######Logestic Regression on Relation - Gender ###########################################################
# Relation File Preprocessing - Train and Text 
likeid_length = len(relation_train_df.like_id)
userid_length = len(relation_train_df.userid)

dictuserid={}
for x in enumerate(train_df['userid'].tolist()):
    val, key = x
    dictuserid[key]=val

dictlikeid={}
for x in enumerate(relation_train_df['like_id'].unique().tolist()):
    val, key = x
    dictlikeid[key]=val

relation_train_df["new_userid"] = relation_train_df["userid"].map(dictuserid)
relation_train_df["new_likeid"] = relation_train_df["like_id"].map(dictlikeid)
row = np.array(relation_train_df["new_userid"])
col = np.array(relation_train_df["new_likeid"])
data = np.fromiter((1 for i in range(likeid_length)), dtype="int")

result = sparse.coo_matrix((data, (row, col)))
no_of_cols = result.shape[1]
total_sum = result.sum(axis=0).tolist()
selected_cols = total_sum[0]
new_cols = []
count = 0
while(count<no_of_cols):
    thissum = selected_cols[count]
    if(thissum>=10 and thissum<=2000):
        new_cols.append(count)
    count = count + 1

userid_col = []

for key in dictuserid.keys():
    userid_col.append(key)

result = result.tocsc()
result_sparse =  sparse.csc_matrix((9500,0))
for each in new_cols:
    this_col = result.getcol(each)
    result_sparse = sparse.hstack([result_sparse,this_col])

result_df = pd.DataFrame(result_sparse.todense(),columns=new_cols)
#X_gender_train, X_gender_test, y_gender_train, y_gender_test = train_test_split(result_df, train_df["gender"], test_size=1500, train_size=8000, random_state=1)

dict_test_users = {}
for x in enumerate(test_df['userid'].tolist()):
    val, key = x
    dict_test_users[key]=val

relation_test_df["new_userid"] = relation_test_df["userid"].map(dict_test_users)
relation_test_df["new_likeid"] = relation_test_df["like_id"].map(dictlikeid)
relation_test_df = relation_test_df.fillna(len(relation_test_df.like_id))
relation_test_df["new_likeid"] = relation_test_df["new_likeid"].astype(int)
row_test = np.array(relation_test_df["new_userid"])
col_test = np.array(relation_test_df["new_likeid"])
data_test = np.fromiter((1 for i in range(len(relation_test_df.like_id))), dtype="int")

result_test = sparse.coo_matrix((data_test, (row_test, col_test)))
columns_test = result_test.col
columns_test = columns_test.tolist()

result_test = result_test.tocsc()
test_result = sparse.csc_matrix((len(test_df["userid"]),0))

for each in new_cols:
    this_col = result_test.getcol(each)
    test_result = sparse.hstack([test_result,this_col])

result_test_df = pd.DataFrame(test_result.todense(),columns=new_cols)

####GENDER USING BAGGING CLASSIFIER######## 

seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
logreg = LogisticRegression()
n_estimators = 5
model_1 = BaggingClassifier(base_estimator=logreg, n_estimators=n_estimators, random_state=seed)
# results_1 = model_selection.cross_val_score(model_1, result_df,train_df["gender"], cv=kfold)
# print("Logistic Regression on Relation: ",results_1.mean())

model_1.fit(result_df,train_df["gender"])
y_predicted = model_1.predict(result_test_df)
test_df["gender"] = y_predicted.astype(int)

#########Multilayer Perceptron on Relation - Age##########################################################
X_age_train, X_age_test, y_age_train, y_age_test = train_test_split(result_df, train_df["age"], test_size=1500, train_size=8000, random_state=1)
X_age_train = X_age_train.values
X_age_test = X_age_test.values

def classifyage(age):
    if(age<=24):
        return 0
    elif(age>=25 and age<=34):
        return 1
    elif(age>=35 and age<=49):
        return 2
    else:
        return 3

age_train = y_age_train.apply(classifyage)
age_test = y_age_test.apply(classifyage)

# One hot encoding of the project
age_train = age_train.values
age_test = age_test.values
age_train = np_utils.to_categorical(age_train)
age_test = np_utils.to_categorical(age_test)
num_classes = age_test.shape[1] 
# print(num_classes)

# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(500, input_dim=len(new_cols), kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# build the model
model = baseline_model()
# Fit the model
model.fit(X_age_train, age_train, epochs=10, batch_size=200,verbose=0)
# Final evaluation of the model
scores = model.evaluate(X_age_test, age_test,verbose=0)
# print("Baseline Error: %.2f%%" % (100-scores[1]*100))
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
y_predicted = model.predict_classes(result_test_df.values)
#print(y_predicted)
#print(len(y_predicted))
test_df["age"] = y_predicted 

########Performing Linear regression to find personality variables########################################
characters = ['age','gender','ope','ext','con','agr','neu']
LIWC_features = [x for x in train_data.columns.tolist()[:] if not x in characters]
LIWC_features.remove('userid')

def predict_personality(feature):
    #Spliting data to train and test - Regression model
    X_personality = train_data[LIWC_features]
    Y_personality = train_data[feature]

    X_train_personality, X_test_personality, Y_train_personality, Y_test_personality = train_test_split(X_personality, Y_personality, test_size=1500, train_size = 8000)

#apply linear regression for all five traits
    linreg = LinearRegression()
    linreg.fit(X_train_personality, Y_train_personality)
    Y_pred = linreg.predict(X_test_personality)
    #print("RMSE "+feature+":", np.sqrt(metrics.mean_squared_error(Y_test_personality, Y_pred)))
    YPRED = linreg.predict(test_data[LIWC_features])
    test_df[feature] = pd.DataFrame(YPRED)
 
personality =  ['ope','ext','con','agr','neu']

for each in personality:
    predict_personality(each)


#print(test_data.head())
#print(test_df.head())

#Creating XML
def createxml(thisID, age, gender, ope, con, ext, agr, neu):
    user = ET.Element("user")
    user.attrib["id"] = thisID    
    if (age==0):
        user.attrib["age_group"] = "xx-24"
    elif(age==1):
        user.attrib["age_group"] = "25-34"
    elif(age==2):
        user.attrib["age_group"] = "35-49"
    else:
        user.attrib["age_group"] = "50-xx"
    if(gender == 1.0):
        user.attrib["gender"] = "female"
    else:
        user.attrib["gender"] = "male"
    user.attrib["extrovert"] = str(round(ext,3))
    user.attrib["neurotic"] = str(round(neu,3))
    user.attrib["agreeable"] = str(round(agr,3))
    user.attrib["conscientious"] = str(round(con,3))
    user.attrib["open"] = str(round(ope,3))    
    tree = ET.ElementTree(user)
    filename = thisID+".xml"
    fullpath = save.join(Outputpath, filename)
    tree.write(fullpath)

for index, row in test_df.iterrows():
    createxml(row['userid'],row['age'],row['gender'],row['ope'],row['con'],row['ext'],row['agr'],row['neu'])


