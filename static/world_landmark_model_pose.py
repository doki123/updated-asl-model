from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics 
import pandas as pd
import joblib 


df = pd.read_csv('static/hand_notorso.csv', index_col=[0])  # removes issue of 'Unnamed: 0' by treating it as index
print(df.shape)

x = df.drop(['CLASS', 'STR_CLASS', 'MOVING', 'TORSO'], axis=1)

y = df.CLASS

print(x.shape)
print(x)

x_train, x_test, y_train, y_test = train_test_split(x.values, y, random_state=4, train_size = 0.8, shuffle=True)


model = LogisticRegression(max_iter=1400, solver="sag")  # note: for higher datasate numbers, saga might be needed instead of sag

model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print(y_pred)

accuracy = metrics.accuracy_score(y_test, y_pred)
print(accuracy)

joblib.dump(model, 'static/world_landmark_model_hand.joblib')


# THIS IS FOR POSE

print('POSE TIME')

df_pose = pd.read_csv('static/hand_withtorso.csv', index_col=[0])  # removes issue of 'Unnamed: 0' by treating it as index
print(df_pose.shape)

x_pose = df_pose.drop(['CLASS', 'STR_CLASS', 'MOVING', 'TORSO'], axis=1)

y_pose = df_pose.CLASS

print(x_pose.shape)
print(x_pose)

x_train, x_test, y_train, y_test = train_test_split(x_pose.values, y_pose, random_state=4, train_size = 0.8, shuffle=True)


model_pose = LogisticRegression(max_iter=1400, solver="sag")  # note: for higher datasate numbers, saga might be needed instead of sag

model_pose.fit(x_train, y_train)
y_pred = model_pose.predict(x_test)

print(y_pred)

accuracy = metrics.accuracy_score(y_test, y_pred)
print(accuracy)

joblib.dump(model_pose, 'static/world_landmark_model_torso.joblib')
