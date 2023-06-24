

import streamlit as st
import numpy as np
import joblib
import json
from streamlit_shap import st_shap
import sklearn.metrics as metrics
import shap
from sklearn.model_selection import train_test_split
import xgboost
import pandas as pd
import sklearn
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import time
from streamlit_option_menu import option_menu
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve

#creating sidebar code
st.set_page_config(page_title=" Bank Customer Segmentation", layout="wide")
with st.sidebar:
    
    selected = option_menu('TERM DEPOSIT PREDICTION APP',
                          ['Information',
                           'Explonatory Data Analysis',
                           'Term Deposit Prediction',
                           'Evaluation Metrics and Plots',
                           'Feature Importance'],default_index=0)
    
if (selected == 'Information'):  
    st.markdown("<h1 style='text-align: center; color: orange;'>CUSTOMER IDENTIFICATION</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(' ')

    with col2:
        st.image("https://www.interest.co.nz/sites/default/files/feature_images/term-deposit.jpg")

    with col3:
        st.write(' ')
    st.markdown("This is a Streamlit deployment of a customer identification model to predict which customers can take bank deposit.The goal of the project is to classify the customer based on four segments: Features that are described as below this page.")
    data = pd.read_csv('/Users/goncaavcicakmak/Desktop/ads-542-finalproject/data/df_final.csv', index_col=0)
    st.markdown("<h3 style='text-align: center; color: orange;'>SHOW DATA</h3>", unsafe_allow_html=True)
    st.write(data)

    st.markdown("<h3 style='text-align: center; color: orange;'>DETAILED USER INPUT EXPLANATION</h3>", unsafe_allow_html=True)

    with open("//Users/goncaavcicakmak/Desktop/ads-542-finalproject/src/descriptions.json") as f:
        segment_descriptions = json.load(f)

    segment_descriptions = pd.DataFrame(segment_descriptions.values(), index=segment_descriptions.keys(), columns=["description"])
    with st.expander('Click to read more about data'):
        st.table(segment_descriptions)

    st.markdown("These descriptions are by no means definitive. Feel free to check it out from https://www.kaggle.com/datasets/sahistapatel96/bankadditionalfullcsv!")


if (selected == 'Explonatory Data Analysis'):
    import plotly.express as px
    st.markdown("<h3 style='text-align: center; color: orange;'>EXPLONATORY DATA ANALYSIS</h3>", unsafe_allow_html=True)
    df = pd.read_csv('/Users/goncaavcicakmak/Desktop/ads-542-project/train.csv')
    fig = px.bar(
        df,
        x= "y",
        y= "age",
        color= "marital",
    )
    fig1 = px.scatter(
        df,
        x= "job",
        y= "age",
        color= "education",
    )
    fig2 = px.bar(
        df,
        x= 'y',
        y= "age",
        color= "job",
    )
    fig3 = px.bar(
        df,
        x= "campaign",
        y= "previous",
        color= 'y',
    )

    fig4= px.bar(
        df,
        x= "previous",
        y= 'y',
        color= "contact",
    )
    fig5 = px.bar(
        df,
        x= "campaign",
        y= 'y',
        color= "poutcome",
    )
    fig6 = px.scatter(
        df,
        x= "nr.employed",
        y= "emp.var.rate",
        color= 'y',
    )
    fig7= px.scatter(
        df,
        x= "campaign",
        y= "pdays",
        color= 'y',
    )
    fig8= px.scatter(
        df,
        x= "cons.price.idx",
        y= "age",
        color= "education",
    )
    fig9 = px.bar(
        df,
        x= "cons.price.idx",
        y= "age",
        color= "education",
    )
    fig10 = px.bar(
        df,
        x= "cons.price.idx",
        y= "cons.conf.idx",
        color= 'y',
    )
    tab1, tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9,tab10,tab11= st.tabs(["Age-Marital-y", "Job-Age-Education", "Age-Job-y","Campaign-Previous-y", "Previous-Contact-y", "Campaign-Poutcome-y","Nr.empoleyed-emp.var.rate-y", "Campaing-pdays-y", "Cons.price.idx-age-education","Cons.price.idx-cons.conf.idx-y", "cons.conf.idx-cons.price.idx-y"])
    with tab1:
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    with tab2:
        st.plotly_chart(fig1, theme=None, use_container_width=True)
    with tab3:
        st.plotly_chart(fig2, theme="streamlit", use_container_width=True)
    with tab4:
        st.plotly_chart(fig3, theme=None, use_container_width=True)
    with tab5:
        st.plotly_chart(fig4, theme="streamlit", use_container_width=True)
    with tab6:
        st.plotly_chart(fig5, theme="streamlit", use_container_width=True)
    with tab7:
        st.plotly_chart(fig6, theme="streamlit", use_container_width=True)
    with tab8:
        st.plotly_chart(fig7, theme=None, use_container_width=True)
    with tab9:
        st.plotly_chart(fig8, theme="streamlit", use_container_width=True)
    with tab10:
        st.plotly_chart(fig9, theme=None, use_container_width=True)
    with tab11:
        st.plotly_chart(fig10, theme=None, use_container_width=True)   

if (selected == 'Term Deposit Prediction'):

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

    model= joblib.load("/Users/goncaavcicakmak/Desktop/ads-542-finalproject/notebooks/xgb_small_model.joblib")

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

        cols = pd.DataFrame([{'age':age, 'job':job1, 'marital':marital1,'education':education1,'campaign':campaign,'pdays':pdays, 'previous':previous, 'poutcome':poutcome1,
                            'emp.var.rate': emp_var_rate,  'cons.price.idx': cons_price_idx, 'cons.conf.idx': cons_conf_idx,
                            'euribor3m':euribor3m,'nr.employed':nr_employed, 'contact':contact1,'default': default1, 'housing': housing1,'loan':loan1}], index=[0])
        
        prediction = model.predict(cols)
        if prediction[0] == 1: 
            st.success('This person can take bank deposit :thumbsup:')
        else: 
            st.error('This person do not take bank deposit  :thumbsdown:') 

    trigger = st.button('Predict', on_click=predict)

if (selected == 'Evaluation Metrics and Plots'):
    st.markdown("<h2 style='text-align: center; color: orange;'>EVALUATION METRICS AND PLOTS</h2>", unsafe_allow_html=True)
    import matplotlib.pyplot as plt
    import seaborn as sns
    y= pd.read_csv("/Users/goncaavcicakmak/Desktop/ads-542-finalproject/data/y_oversampled.csv", index_col=[0])
    X= pd.read_csv("/Users/goncaavcicakmak/Desktop/ads-542-finalproject/data/X_oversampled.csv", index_col=[0])
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

    modelk = KNeighborsClassifier(n_neighbors=3, weights= 'distance')
    modelk.fit(x_train, y_train)
    accuracy = modelk.score(x_test, y_test)
    y_pred = modelk.predict(x_test)
    class_names=['yes','no']
    st.write("Accuracy: ", accuracy.round(2))
    st.write("Precision: ", precision_score(y_test, y_pred, labels = class_names).round(2))
    st.write("Recall: ", recall_score(y_test, y_pred, labels = class_names).round(2))

    col1, col2 = st.columns(2)
    with col1:
        st.header("CONFUSION MATRIX")
        cm=confusion_matrix(y_test, y_pred)
        fig,ax=plt.subplots(figsize=(8,8))
        sns.heatmap(cm, ax=ax, annot=True)
        st.write(fig,width=600, height=600)


    with col2:
        st.header("ROC CURVE")
        probs = modelk.predict_proba(x_test)
        preds = probs[:,1]
        fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
        roc_auc = metrics.auc(fpr, tpr)
        fig,ax=plt.subplots(figsize=(8,8))
        ax.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        ax.set_ylabel('FPR')
        ax.set_xlabel('TPR')
        st.pyplot(fig,width=600, height=600)


if (selected == 'Feature Importance'): 
    y= pd.read_csv("/Users/goncaavcicakmak/Desktop/ads-542-finalproject/data/y_oversampled.csv", index_col=[0])
    X= pd.read_csv("/Users/goncaavcicakmak/Desktop/ads-542-finalproject/data/X_oversampled.csv", index_col=[0])

    @st.cache_data
    def load_model(X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
        d_train = xgboost.DMatrix(X_train, label=y_train)
        d_test = xgboost.DMatrix(X_test, label=y_test)
        params = {'max_depth': 17,
                'learning_rate': 0.04253736068327828,
                'n_estimators': 517,
                'subsample': 0.6489298539237697,
                'colsample_bytree': 0.5117164405901318,
                'reg_alpha': 0.3042725884752507,
                'reg_lambda': 0.3474868393613407}
        model_xg = xgboost.train(params, d_train, 10, evals = [(d_test, "test")], verbose_eval=100, early_stopping_rounds=20)
        return model_xg

    st.markdown("<h3 style='text-align: center; color: orange;'>FEATURE IMPORTANCE WITH SHAP</h3>", unsafe_allow_html=True)

    X_display,y_display = X,y

    model_xg = load_model(X, y)

    # compute SHAP values
    explainer = shap.Explainer(model_xg, X)
    shap_values = explainer(X)

    explainer1 = shap.TreeExplainer(model_xg)
    shap_values1 = explainer1.shap_values(X)

    tab01,tab02,tab03,tab04= st.tabs(['WATREFALL','BEESWARM','FORCE PLOT','FORCE PLOT'])
    with tab01:
        st_shap(shap.plots.waterfall(shap_values[0]), height=800, width=800)
    with tab02:
        st_shap(shap.plots.beeswarm(shap_values), height=800, width=800)
    with tab03:
        st_shap(shap.force_plot(explainer1.expected_value, shap_values1[0,:], X_display.iloc[0,:]), height=800, width=1000)

    with tab04:
        st_shap(shap.force_plot(explainer1.expected_value, shap_values1[:1000,:], X_display.iloc[:1000,:]), height=600, width=1000)






