import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
import seaborn as sns
#Bibliotheksimport zur Warnungssteuerung
import warnings
warnings.filterwarnings("ignore")
#Instalacion e importacion de libreria de Cryptowatch
import logging
import cryptowatch as cw
#API-SCHLÜSSEL erforderlich, um die Cryptowatch-Bibliothek zu verwenden
cw.api_key = "BM0DRS9QDJ4W7JCL76PD"

logging.basicConfig()
logging.getLogger("cryptowatch").setLevel(logging.WARNING)

cw.exchanges.list()

#Wählen Sie Exchange und CryptoActive zur Analyse aus
MARKET ="BINANCE"
TICKET="etheur"

#Sehen Sie sich die verschiedenen Cryptoactives an, die in der ausgewählten Börse verfügbar sind
list = []
selected = cw.markets.list(MARKET)
for market in selected.markets:
    list.append(market.pair)
df_list = pd.DataFrame (list, columns = ['Währung'])
print(df_list)

#Wir wählen das Ticket (Paar der ausgewählten Kryptoaktiven) aus, von dem wir alle seine Daten erhalten möchten
candles = cw.markets.get(MARKET+":"+TICKET, ohlc=True, periods=["1m","15m","4h", "1h", "1d", "1w"])


# Wir generieren den Datenrahmen mit den Daten
rows_list = []
for x in candles.of_1d:
    close_ts = datetime.fromtimestamp(x[0])
    open_value = x[1]
    high_value = x[2]
    low_value = x[3]
    close_value = x[4]
    volume_base = x[5]
    volume_quote = x[6]
    rows_list.append([TICKET,close_ts , open_value , high_value , low_value ,close_value ,volume_base ,volume_quote])
df = pd.DataFrame(rows_list,columns = ["ticket","close_ts" , "open_value" , "high_value" , "low_value" ,"close_value" ,"volume_base" ,"volume_quote"])
df.to_csv('eth.csv', index=False)
df2 = pd.read_csv("eth.csv")
print(df2)