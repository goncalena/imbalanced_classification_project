

import streamlit as st
import numpy as np
import joblib
import json
import pandas as pd
import sklearn
from PIL import Image
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title=" Bank Customer Segmentation", layout="wide")
st.title("Customer Identification")
st.markdown("This is a Streamlit deployment of a customer identification model to predict which customers can take bank deposit.The goal of the project is to classify the customer based on four segments: Features that are described as below this page.")

col1, col2, col3 = st.columns(3)

# Set up Streamlit structure
with col1:
    st.subheader("Bank Client Data")
    age = st.number_input(label='Age:', min_value = 17.0,max_value = 99.0,step = 1.0,format="%.1f")
    marital = st.selectbox('Sex', ['married', 'single','divorced','unknown'])
    job = st.selectbox("Job", ["admin", "bluecollar" "technician","services","management", "retired","entrepreneur","selfemployed", "housemaid", "unemployed", "student"])
    education = st.selectbox("Education Degree", ['universitydegree', 'highschool', 'basic9y', 'professionalcourse','basic.y', 'basic6y', 'illiterate'])
    default =  st.selectbox("Have any loan in Default", ['yes','no'])
    loan =  st.selectbox("Have you any personal loan?", ['yes','no'])
    housing =  st.selectbox("Have any housing loan?", ['yes','no'])

with col2:
    st.subheader("Other Attributes")
    contact = st.selectbox("Which type of device is used for communication?", ['telephone', 'cellular'])
    campaign = st.number_input(label="campaign", min_value = 1.0,max_value = 37.0 ,step = 1.0,format="%.1f")
    poutcome = st.selectbox("what is outcome?",['nonexistent', 'failure', 'success'])
    previous = st.number_input(label="previous", min_value = 0.0,max_value = 7.0 ,step = 1.0,format="%.1f")
    pdays = st.number_input(label="pdays", min_value = 0.0,max_value = 20.0 ,step = 1.0,format="%.1f")

with col3:
    st.subheader("Social and Economic Context Attributes")
    emp_var_rate = st.number_input(label = 'emp_var_rate', min_value = -3.5,max_value = 1.4 ,step = 0.1,format="%.1f")
    cons_price_idx = st.number_input(label="cons.price.idx",min_value = 92.000,max_value = 95.000 ,step = 0.001,format="%.3f")
    cons_conf_idx = st.number_input(label="cons.conf.idx",min_value = -51.0,max_value = -26.0 ,step = 0.1,format="%.1f")
    nr_employed = st.number_input(label="nr.employed",min_value = 4900.000,max_value = 5250.0000 ,step = 0.001,format="%.3f")
    euribor3m = st.number_input(label="euribor3m",min_value = 0.600,max_value = 6.000 ,step = 0.001,format="%.3f")

model= joblib.load("/Users/goncaavcicakmak/Desktop/ads-542-finalproject/notebooks/xgb_model.joblib")

def predict(): 
    if marital== 'married':
         marital1=1
    elif marital=='single':
         marital1=2
    elif marital=='divorced':
         marital1=0
    else:
         marital1=3
    
    if poutcome=='nonexistent':
         poutcome1=1
    elif poutcome=='failure':
         poutcome1=0
    else:
         poutcome1=2
    
    if job=='admin':
         job1=0
    elif job=='bluecollar':
         job1=1
    elif job=='technician':
         job1=9
    elif job=='services':
         job1=7
    elif job=='management':
         job1=4
    elif job=='retired':
         job1=5
    elif job=='entrepreneur':
        job1=2
    elif job=='selfemployed':
         job1=6
    elif job=='housemaid"':
         job1=3
    elif job=='unemployed':
         job1= 10      
    else:
         job1=8

    if education=='universitydegree':
         education1=6
    elif education=='highschool':
         education1=3
    elif education=='basic9y':
         education1=2
    elif education=='professionalcourse':
         education1=5
    elif education=='basic.y':
         education1=0
    elif education=='basic6y':
         education1=1
    else:
         education1=4

    if contact=='telephone':
         contact1=1
    else:
         contact1=0

    if housing =='yes':
         housing1=1
    else:
         housing1=0

    if default=='yes':
         default1=1
    else:
         default1=0

    if loan=='yes':
         loan1=1
    else:
         loan1=0

    cols = pd.DataFrame([{'education':education1, 'job':job1, 'contact':contact1,'default': default1, 'housing': housing1,
           'loan':loan1,'age':age,'marital':marital1, 'campaign':campaign,'pdays':pdays,'previous':previous,
           'poutcome':poutcome1, 'emp.var.rate': emp_var_rate , 'cons.price.idx': cons_price_idx, 
           'cons.conf.idx': cons_conf_idx, 'euribor3m':euribor3m, 'nr.employed':nr_employed }], index=[0])
    
    prediction = model.predict(cols)
    if prediction[0] == 1: 
        st.success('This person can take bank deposit :thumbsup:')
    else: 
        st.error('This person do not take bank deposit  :thumbsdown:') 

trigger = st.button('Predict', on_click=predict)

st.subheader("User Input for Segment Prediction")
with open("/Users/goncaavcicakmak/Desktop/adas-542-final-proje/notebooks/descriptions.json") as f:
    segment_descriptions = json.load(f)

segment_descriptions = pd.DataFrame(segment_descriptions.values(), index=segment_descriptions.keys(), columns=["description"])
st.table(segment_descriptions)

st.markdown("These descriptions are by no means definitive. Feel free to check it out from https://www.kaggle.com/datasets/sahistapatel96/bankadditionalfullcsv!")

image = Image.open('/Users/goncaavcicakmak/Desktop/ads-542-finalproject/src/snail.png')
st.image(image)

st.markdown("For more information about the project e.g. how it was trained visit....")
