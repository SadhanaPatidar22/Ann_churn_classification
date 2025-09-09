import tensorflow as tf
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
import pickle

## load the trained model
model = tf.keras.models.load_model('model.h5')

## load the label encoder
with open('label_encoder_gen.pkl','rb') as file:
    label_encoder_gen = pickle.load(file)

## load the one hot  encoder
with open("onehot_encoder_geo.pkl",'rb') as file:
    onehot_encoder_geo = pickle.load(file)

# load the scaler
with open("scaler.pkl",'rb') as file:
    scaler = pickle.load(file)


# title
st.title("Customer Churn Prediction")

# user input
geography = st.selectbox('Geography',onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender",label_encoder_gen.classes_)
age = st.slider('Age',18,92)
balance = st.number_input("Balance")
credit_score = st.number_input("CreditScore")
estimated_salary = st.number_input("EstimatedSalary")
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('NumOfProducts',1,4)
has_cr_card = st.selectbox("Has Credit Card",[0,1])
is_active_member = st.selectbox('Is active number',[0,1])

input_data = pd.DataFrame({
    'CreditScore' : [credit_score],
    'Gender' : [label_encoder_gen.transform([gender])[0]],
    'Age' : [age],
    'Tenure' :[tenure],
    'Balance' : [balance],
    'NumOfProducts' :[num_of_products],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember' : [is_active_member],
    'EstimatedSalary' : [estimated_salary]
})

#one hot encoding on geography
geo_encode = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encode_df = pd.DataFrame(geo_encode,columns = onehot_encoder_geo.get_feature_names_out(["Geography"]))

## combine input data and geo_encode_df
input_data = pd.concat([input_data.reset_index(drop=True),geo_encode_df])

input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]
st.write(f"Churn Probability : {prediction_proba:.2f}")

if prediction_proba>0.5 :
    st.write("customer is likely to churn.")
else:
    st.write("customer is not likely to churn.")


