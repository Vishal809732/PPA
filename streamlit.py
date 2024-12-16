
#pip install streamlit
#pip install --upgrade protobuf
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
st.title("Hello Streamlit")

st.header("Header")
st.subheader("Sub Header")
st.text("The quick brown fox jumps over the lazy dog")
st.markdown("""# H1 Tag
            #H2 Tag
            #H3 Tag
            :sunglasses:
            Bold: bold
            Italic:_ italics _
            ** bold ** <br> _ italics _
            """,True)

df=pd.read_csv("kc_house_data.csv")   
df=df.sample(frac=0.2) 
st.dataframe(df)        

st.dataframe(df,width=300,height=200)   
#df1 = pd.DataFrame({'x': [1, 2, 3], 'y': [10, 30, 70]})
st.set_option('deprecation.showPyplotGlobalUse', False)
sns.lineplot(x='sqft_living', y='price', data=df)
st.pyplot()

sns.boxplot(x='bedrooms',y='price',data=df)
st.pyplot()

sns.lineplot(x='bedrooms',y='price',data=df)
st.pyplot()

st.map()

st.image("wallpaper_analytics.jpg")
st.audio("sample.mp3")
st.video("https://www.youtube.com/watch?v=wlkCQXHEgjA")

#widgets
#Button returns boolean value T/F
if st.button("Buy"):
    st.write("Purchased")

#text input
    
name=st.text_input("Enter Name")
st.write("Name is",name)

#text area input
address=st.text_area("Enter Address")
st.write("Address is:",address)

#date input
date=st.date_input("Enter a date:")

#time input
time=st.time_input("Enter a time:")

#Checkbox input
if st.checkbox("I accept the conditions",value=True):
    st.write("Thanks")

radio_var=st.radio("Colors=",['red','green','blue'],index=0)

sel_box_var=st.selectbox("Colors=",['red','green','blue'],index=0)

st.write(radio_var,sel_box_var)

multi_var=st.multiselect("Colors=",['red','green','blue'])

st.write(multi_var)

st.slider("age",min_value=18,max_value=60,value=30,step=2)

st.number_input("Enter a number",min_value=18.0,max_value=60.0,value=30.0,step=2.0)






