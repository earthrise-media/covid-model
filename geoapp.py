import streamlit as st
import numpy as np
import pandas as pd
import random

val = st.slider(
	'Time period',
	0, 9, 1
)

df = pd.read_csv("data/unemployment.tsv", sep='\t')

for i in range(10):
	varname = "rate-%s" % i
	df[varname] = df["rate"] + random.uniform(0.1, 0.2)

values = df.to_dict('records')
iteration_varname = "rate-%s" % val


@st.cache(suppress_st_warning=True)
def show(x):
	st.vega_lite_chart(
		{
			"$schema": "https://vega.github.io/schema/vega-lite/v4.json",
			"data": {
				"url": "https://raw.githubusercontent.com/earthrise-media/covid-model/master/data/us-10m.json",
				"format": {
					"type": "topojson",
					"feature": "counties"
				}
			},
			"transform": [{
				"lookup": "id",
				"from": {
					"data": {
						"values": values
					},
					"key": "id",
					"fields": [x]
				}
			}],
			"projection": {
				"type": "albersUsa"
			},
			"mark": "geoshape",
			"encoding": {
				"color": {
					"field": x,
					"type": "quantitative"
				}
			},
			"config": {
	            "style": {
	                "cell": {
	                    "strokeOpacity": 0
	                } 
	            }
	        }
	    },
	    use_container_width=True
	)

show(iteration_varname)