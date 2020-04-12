# covid-model
Compartmental model with age cohorts

## setup

The web app relies on [Python 3](https://www.python.org/downloads) and [Streamlit](https://www.streamlit.io).

```bash
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```

When finished:

```bash
deactivate
```

## run

You can either run the app locally or from the latest (or previous) GitHub commit, just to check things out quickly.  

```
streamlit run https://raw.githubusercontent.com/earthrise-media/covid-model/master/app.py
```
Or from within the local directory:
```
streamlit run app.py
```
The browser will open with the Streamlit app.

## developing
There is an option in Streamlit to reload upon saving. The best setup that I've discovered so far is to run the code editor and browser in dual panes, fullscreen.

## deploy

**TODO**
