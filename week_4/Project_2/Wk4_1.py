# John Allen
# Diana Walker
# frank Perez
# Michael Cassano

import requests
import json
from openpyxl import Workbook

# API URLs
cities_url = "https://cityweatherclass.free.beeceptor.com/cities"
weather_url = "https://cityweatherclass.free.beeceptor.com/weather"

# Get data
cities_data = json.loads(requests.get(cities_url).text)
weather_data = json.loads(requests.get(weather_url).text)

# Combine by city
combined = []
city_lookup = {c["city"]: c for c in cities_data}

for w in weather_data:
    if w["city"] in city_lookup:
        c = city_lookup[w["city"]]
        combined.append({
            "city": w["city"],
            "region": c["region"],
            "population": c["population"],
            "temperature": w["temperature"],
            "humidity": w["humidity"]
        })

# Save to Excel
wb = Workbook()
ws = wb.active
ws.title = "City Weather Report"
ws.append(["City", "Region", "Population", "Temperature", "Humidity"])

for r in combined:
    ws.append([r["city"], r["region"], r["population"], r["temperature"], r["humidity"]])

wb.save("city_weather_report.xlsx")
print("Done! File saved: city_weather_report.xlsx")