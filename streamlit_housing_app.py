# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 07:48:58 2021

@author: manis
"""

#https://raw.githubusercontent.com/Shreyas3108/house-price-prediction/master/kc_house_data.csv
#https://www.youtube.com/watch?v=_dzr1pm3Ymw
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
st.title('DreamHomes.com')
st.text('Find you dream homes here')
df=pd.read_csv('kc_house_data.csv')
st.image('house.jpg')
st.video('https://www.youtube.com/watch?v=_dzr1pm3Ymw')
price_set=st.slider("Price Range",min_value=int(df['price'].min()),max_value=int(df['price'].max()),step=50,value=int(df['price'].max()))
st.text("Price Selected is "+str(price_set))
fig=px.scatter_mapbox(df.loc[df['price']<price_set],lat='lat',lon='long',color='sqft_living',size='price')
fig.update_layout(mapbox_style='open-street-map')
st.plotly_chart(fig)
st.header("Price Predictor")
sel_box_var=st.selectbox("Select Method",['Linear','Ridge','Lasso'],index=0)
multi_var=st.multiselect("Select Additional Variables for Accuracy=",['sqft_living','sqft_lot','sqft_basement'])
df_new=[]
df_new=df[multi_var]
if sel_box_var=='Linear':
    df_new['bedrooms']=df['bedrooms']
    df_new['bathrooms']=df['bathrooms']
    X=df_new
    Y=df['price']
    model=LinearRegression()
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
    reg=model.fit(X_train,Y_train)  
    Y_pred = reg.predict(X_test)
    st.text("Intercept="+str(reg.intercept_))
    st.text("Coefficient="+str(reg.coef_))
    st.text("R^2="+str(r2_score(Y_test,Y_pred)))
elif sel_box_var=='Lasso':
    df_new['bedrooms']=df['bedrooms']
    df_new['bathrooms']=df['bathrooms']
    X=df_new
    Y=df['price']
    model=Lasso()
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
    reg=model.fit(X_train,Y_train)  
    Y_pred = reg.predict(X_test)
    st.text("Intercept="+str(reg.intercept_))
    st.text("Coefficient="+str(reg.coef_))
    st.text("R^2="+str(r2_score(Y_test,Y_pred)))
else:
    df_new['bedrooms']=df['bedrooms']
    df_new['bathrooms']=df['bathrooms']
    X=df_new
    Y=df['price']
    model=Ridge()
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
    reg=model.fit(X_train,Y_train)  
    Y_pred = reg.predict(X_test)
    st.text("Intercept="+str(reg.intercept_))
    st.text("Coefficient="+str(reg.coef_))
    st.text("R^2="+str(r2_score(Y_test,Y_pred)))    
st.set_option('deprecation.showPyplotGlobalUse',False)
#sns.regplot(Y_test,Y_pred)
#st.pyplot()
count=0
pred_val=0
for i in df_new.keys():
    try:
        val=st.text_input("Enter no./val of "+i)
        pred_val=pred_val+float(val)*reg.coef_[count]
        count=count+1
    except:
        pass
st.text('Predicted Prices are:'+str(pred_val+reg.intercept_))
st.header("Application Details")
img=st.file_uploader("Upload Application")
st.text("Details for the representative to contact you")
st.text("Enter your address")
address=st.text_area("Your address Here")
date=st.date_input("Enter a date")
time=st.time_input("Enter the time")
if st.checkbox("I confirm the date and time",value=False):
    st.write("Thanks for confirming")
st.number_input("Rate our site",min_value=1.0,max_value=10.0,step=1.0)
