import joblib
import pandas as pd
import streamlit as st


def main():
	html_temp = """
		<div style="background-color: lightblue;padding: 16px">
			<h2 style="color: black";text-align: center>Health Insurance Cost Prediction Using ML</h2> 
		</div>"""	

	st.markdown(html_temp, unsafe_allow_html=True)

	gr = joblib.load("model_")

	p1 = st.slider("Enter your age", 18, 100)

	s1 = st.selectbox("Sex", ("Male", "Female"))
	if s1 == "Male":
		p2 = 1
	else:
		p2 = 0

	p3 =st.number_input("Enter Your BMI Value")
	p4 = st.slider("Enter Number of Children",0,4) 
    
	s2 = st.selectbox("Smoker", ("Yes", "No"))
	if s2 == "Yes":
		p5 = 1
	else:
		p5 = 0

	p6 = st.slider("Enter your region [1-4]", 1, 4)

	if st.button("Predict"):
		data_predict = pd.DataFrame(data={
		    'age' : p1,
		    'sex' : p2,
		    'bmi' : p3,
		    'children' : p4,
		    'smoker' : p5,
		    'region' : p6
		}, index=["v"])

		new_pred = gr.predict(data_predict)
		st.success(f"Medical Insurance cost for New Customer is : {new_pred[0]}")

main()