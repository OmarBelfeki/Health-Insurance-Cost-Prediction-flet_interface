import flet as ft

import pandas as pd
import joblib

gr = joblib.load("model_")


def main(page: ft.Page):
	page.window_height = 700
	page.window_width = 400
	page.title = "Health Insurance Cost Prediction"

	def predict(e):
		age_ = int(age.value)
		sex_ = int(gen.value)
		bmi_ = float(bmi.value)
		children_ = int(children.value)
		smoker_ = int(smok.value)
		if reg.value == "southwest":
			region_ = 1
		if reg.value == "southeast":
			region_ = 2
		if reg.value == "northwest":
			region_ = 3
		if reg.value == "northeast":
			region_ = 4

		data_predict = pd.DataFrame(data={
		    'age' : age_,
		    'sex' : sex_,
		    'bmi' : bmi_,
		    'children' : children_,
		    'smoker' : smoker_,
		    'region' : region_
		}, index=["v"])

		new_pred = gr.predict(data_predict)
		label.value = f"Medical Insurance cost for New Customer is : {new_pred[0]}"

		page.update()

	title = ft.Text(value="Health Insurance Cost Prediction", size=20, color="red", weight="bold")
	age = ft.TextField(label="Age")
	sex = ft.Row(
		controls=[
			ft.Text(value="Sex"),
			gen:=ft.RadioGroup(
				content=ft.Row(
					#alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
					controls=[
						ft.Radio(value="0", label="female"),
						ft.Radio(value="1", label="male"),
					]
				)
			)
		]
	)
	bmi = ft.TextField(label="BMI")
	children = ft.TextField(label="children?")
	smoker = ft.Row(
		controls=[
			ft.Text(value="Smoker"),
			smok:=ft.RadioGroup(
				content=ft.Row(
					#alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
					controls=[
						ft.Radio(value="0", label="no"),
						ft.Radio(value="1", label="yes"),
					]
				)
			)
		]
	)
	region = ft.Row(
		controls=[
			ft.Text(value="Region"),
			reg:=ft.Dropdown(
				#label="Region",
				hint_text="Choose Region",
				width=200,
				options=[
					ft.dropdown.Option("southwest"),
					ft.dropdown.Option("southeast"),
					ft.dropdown.Option("northwest"),
					ft.dropdown.Option("northeast")
				]
			)
		]
	)
	btn = ft.ElevatedButton(text="Click", on_click=predict)
	label = ft.Text(color="green", size=20)



	page.add(title, age, sex, children, bmi, smoker, region, btn, label)


ft.app(target=main)