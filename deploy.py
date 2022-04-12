from pyngrok import ngrok 
# !ngrok authtoken 27CLVm8I6NY5yJuk5a3IMDf0708_7wrZyrTXD6eDcLaHRid91
!nohup streamlit run app.py & 


url = ngrok.connect(port = 8501)
url #generates our URL


# !streamlit run --server.port 80 app.py >/dev/null #used for starting our server


# !pip install streamlit


# !pip install pyngrok