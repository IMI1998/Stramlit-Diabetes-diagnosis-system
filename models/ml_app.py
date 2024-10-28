import streamlit as st
import joblib
import os
import numpy as np


label_dict = {"No":0 , "Yes":1}
gender_map = {"Female":0,"Male":1}
target_label_map = {"Negative": 0 , "Positive":1}

def get_fvalue(val):
    feature_dict = {"No":0 , "Yes":1}
    for key , value in feature_dict.items():
        if val == key:
            return value

def get_value(val , my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value


@st.cache
def load_data(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file) , "rb"))
    return loaded_model

def run_ml_app():
    st.subheader("ML Prediction")
   


    col1 , col2 = st.columns(2)

    with col1:
        age = st.number_input("Age" , 10 ,100)
        gender = st.radio("Gender" , ["Female" , "Male"])
        polyuria = st.radio("Polyuria" , ["No" , "Yes"])
        polydipsia = st.radio("Polydipsia" , ["No" , "Yes"])
        sudden_weight_loss = st.selectbox("Sudden_weight_loss" , ["No" , "Yes"])
        weakness = st.radio("weakness" , ["No" , "Yes"])
        polyphagia = st.radio("polyphagia" , ["No" , "Yes"])
        genital_thrush = st.selectbox("Genital_thrush" , ["No" , "Yes"])

    with col2:
         visual_blurring = st.selectbox("Visual_blurring" , ["No" , "Yes"])
         itching = st.radio("itching" , ["No" , "Yes"])
         irritability = st.radio("irritability" , ["No" , "Yes"])
         delayed_healing = st.radio("delayed_healing" , ["No" , "Yes"])
         partial_paresis = st.selectbox("Partial_paresis" , ["No" , "Yes"])
         muscle_stiffness = st.radio("muscle_stiffness" , ["No" , "Yes"])
         alopecia = st.radio("alopecia" , ["No" , "Yes"])
         obesity = st.select_slider("obesity" , ["No" , "Yes"])


    with st.expander("Your Selected Option"):
        result = {
            "age":age,
            "gender":gender,
            "polyuria":polyuria,
            "polydipsia":polydipsia,
            "sudden_weight_loss":sudden_weight_loss,
            "weakness":weakness,
            "polyphagia":polyphagia,
            "genital_thrush":genital_thrush,
            "visual_blurring":visual_blurring,
            "itching":itching,
            "irritability":irritability,
            "delayed_healing":delayed_healing,
            "partial_paresis":partial_paresis,
            "muscle_stiffness":muscle_stiffness,
            "alopecia":alopecia,
            "obesity":obesity,
        } 
        st.write(result)

        encoded_result = []
        for i in result.values():
            if type(i) == int:
                encoded_result.append(i)
            elif i in ["Female" , "Male"]:
                res = get_value(i,gender_map)
                encoded_result.append(res)
            else:
                encoded_result.append(get_fvalue(i))

        st.write(encoded_result)                
   
    with st.expander("Preiction Result"):
        single_sample = np.array(encoded_result).reshape(1,-1)
        st.write(single_sample)

        model = load_data("logistic_regression_model_diabetes_21_oct_2020.pkl")
        prediction = model.predict(single_sample)
        pred_prob = model.predict_proba(single_sample)
        st.write(prediction)
        st.write(pred_prob)

        if prediction == 1:
            st.warning("Positive Risk {}".format(prediction[0]))
            pred_probability_score = {"Negative DM Risk": pred_prob[0][0]*100 , 
            "Positive DM Risk": pred_prob[0][1]*100}
            st.write(pred_probability_score)
        else:
            st.success("Negative Risk {}".format(prediction[0]))
            pred_probability_score = {"Negative DM Risk": pred_prob[0][0]*100 , 
            "Positive DM Risk": pred_prob[0][1]*100}
            st.write(pred_probability_score)
