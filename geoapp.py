import streamlit as st
import numpy as np
import pandas as pd

st.vega_lite_chart(
	{
		"$schema": "https://vega.github.io/schema/vega-lite/v4.json",
		"width": 500,
		"height": 300,
		"data": {
			"url": "https://raw.githubusercontent.com/vega/vega/master/docs/data/us-10m.json",
			"format": {
				"type": "topojson",
				"feature": "counties"
			}
		},
		"transform": [{
			"lookup": "id",
			"from": {
				"data": {
					"url": "https://raw.githubusercontent.com/vega/vega/master/docs/data/unemployment.tsv"
				},
				"key": "id",
				"fields": ["rate"]
			}
		}],
		"projection": {
			"type": "albersUsa"
		},
		"mark": "geoshape",
		"encoding": {
			"color": {
				"field": "rate",
				"type": "quantitative"
			}
		},
		'config': {
            'style': {
                'cell':{
                    'strokeOpacity':0
                } 
            }
        }
    }
)

