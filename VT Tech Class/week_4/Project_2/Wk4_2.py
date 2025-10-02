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

# Save to Excel (Sheet 1: City Weather Report)
wb = Workbook()
ws = wb.active
ws.title = "City Weather Report"
ws.append(["City", "Region", "Population", "Temperature", "Humidity"])

for r in combined:
    ws.append([r["city"], r["region"], r["population"], r["temperature"], r["humidity"]])

# ---- Step 1: Group Cities by Region (only city, temperature, region) ----
regions_order = ["Midwest", "West", "South"]
groups = {reg: [] for reg in regions_order}

for rec in combined:
    reg = rec["region"]
    if reg in groups:  # only keep the three requested regions
        groups[reg].append({
            "city": rec["city"],
            "temperature": rec["temperature"],
            "region": rec["region"]
        })

# ---- Step 2: Create a New Worksheet for Regions ----
ws2 = wb.create_sheet(title="Cities by Region")
ws2.append(["City", "Temperature", "Region"])

for idx, reg in enumerate(regions_order):
    for item in groups[reg]:
        ws2.append([item["city"], item["temperature"], item["region"]])
    # leave a blank row between regions (but not after the last one)
    if idx < len(regions_order) - 1:
        ws2.append([])

# ---- Step 3: Save and Submit ----
wb.save("city_weather_report_updated.xlsx")
print("Done! File saved: city_weather_report_updated.xlsx")