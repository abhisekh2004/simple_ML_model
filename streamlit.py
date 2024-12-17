import streamlit as st
import matplotlib.pyplot as plt 
import pandas as pd
import pickle

regressor_model = pickle.load(open('Models/linearmodel.pkl','rb'))
scaler = pickle.load(open('Models/scaler.pkl','rb'))

def predict_Height(weight):
    weight_scaled = scaler.transform([[weight]])
    res = regressor_model.predict(weight_scaled)
    return res[0]
    
        
def main():
    st.title("Hello Welcome to my Height prediction app")
    wt = st.text_input("ENTER YOUR WEIGHT in kgs","Type here")
    result = ""
    if st.button("Predict"):
        result = predict_Height(wt)
    st.success('The predicted height is {} cm'.format(result))

if __name__ == '__main__':
    main()