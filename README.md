# covid-model
Compartmental model with age cohorts

## setup

The web app is built with [Streamlit](https://www.streamlit.io) and deployed to [Heroku](https://www.heroku.com/) using a [Docker](https://www.docker.com/) container.

Build and run the container from within the top-level project directory:

```bash
docker build -f Dockerfile -t app:latest .
docker run -p 8501:8501 app:latest
```

Navigate to [localhost:8501](http://localhost:8501/) in your browser.

## developing
There is an option in Streamlit to reload upon saving. The best setup that I've discovered so far is to run the code editor and browser in dual panes, fullscreen.

```
streamlit run app.py  # if you don't want to run docker
```

## deploy

Login to the Heroku CLI using the Earthrise Developer credentials:

```
heroku login
```

Then deploy the web app:

```
git push heroku master
```
