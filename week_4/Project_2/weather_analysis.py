# Frank Perez
# Diana Walker
# John Allen
# https://cityweatherclass.free.beeceptor.com/weather 
# https://cityweatherclass.free.beeceptor.com/cities

import requests
import json
from openpyxl import Workbook

wb = Workbook()
ws = wb.active
ws.title = "City Weather Data"

weather_url = "https://cityweatherclass.free.beeceptor.com/weather"
cities_url = "https://cityweatherclass.free.beeceptor.com/cities"

response = dict()

def fetch_data(url):
    response = requests.get(url)
    return json.loads(response.text)

cities_data = fetch_data(cities_url)
print(cities_data)
weather_data = fetch_data(weather_url)
print(weather_data)












wb.save("./week_4/spreadsheets/city_weather_report.xlsx")

