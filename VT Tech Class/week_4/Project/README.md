# Weather Analysis Project

This project demonstrates how to fetch data from multiple APIs, combine datasets, and create professional Excel reports using Python.

## Overview

The project fetches data from two APIs:
- **Cities API**: Provides city names, populations, and regions
- **Weather API**: Provides current temperature and humidity for cities

The data is combined using city names as the key, then formatted into professional Excel spreadsheets with multiple views.

## Project Versions

### Version 1 (W4P1)
- Single worksheet with complete city and weather data
- Basic formatting and analysis
- File: `city_weather_report.xlsx`

### Version 2 (W4P2) - Updated
- **Two worksheets** for different views of the data
- **Sheet 1**: Complete dataset (City, Region, Population, Temperature, Humidity)
- **Sheet 2**: Regional grouping with spacing for easy analysis
- Enhanced data with more cities (15 vs 10)
- File: `city_weather_report_updated.xlsx`

## Features

- ✅ API data fetching with error handling
- ✅ JSON parsing and data validation
- ✅ Data combination using city names as keys
- ✅ Professional Excel formatting with headers, borders, and styling
- ✅ Auto-adjusted column widths
- ✅ Comprehensive logging and progress tracking

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Version 1 (Original)
Run the original script:
```bash
python weather_analysis.py
```

### Version 2 (Updated with Regional Grouping)
Run the updated script:
```bash
python weather_analysis_updated.py
```

This will:
1. Fetch data from both APIs (with fallback to mock data)
2. Combine the datasets
3. Create two worksheets:
   - Complete city weather report
   - Regional grouping with spacing
4. Save it as `city_weather_report_updated.xlsx`

### Analysis Scripts
- `python read_report.py` - Analyze the original report
- `python read_updated_report.py` - Analyze the updated report with both sheets
- `python compare_reports.py` - Compare both versions

## Output

### Version 1 Output
The original script generates `city_weather_report.xlsx` with the following columns:
- **City**: Name of the city
- **Region**: Geographic region of the U.S.
- **Population**: Total population
- **Temperature**: Current temperature in Fahrenheit
- **Humidity**: Current humidity percentage

### Version 2 Output
The updated script generates `city_weather_report_updated.xlsx` with:

**Sheet 1 - City Weather Report:**
- Complete dataset with all 5 columns (same as Version 1)
- 15 cities (expanded from 10)

**Sheet 2 - Cities by Region:**
- **City**: Name of the city
- **Temperature**: Current temperature in Fahrenheit
- **Region**: Geographic region of the U.S.
- Grouped by region with blank rows for readability
- Focused view for regional analysis

## Error Handling

The script includes comprehensive error handling for:
- Network connection issues
- Invalid JSON responses
- Missing data matches
- File writing errors

## Project Structure

```
week_4/Project/
├── weather_analysis.py           # Original script (Version 1)
├── weather_analysis_updated.py   # Updated script (Version 2)
├── read_report.py               # Analyze original report
├── read_updated_report.py       # Analyze updated report
├── compare_reports.py           # Compare both versions
├── project_summary.py           # Learning objectives
├── requirements.txt             # Dependencies
├── README.md                    # This file
├── city_weather_report.xlsx     # Generated report (Version 1)
└── city_weather_report_updated.xlsx  # Generated report (Version 2)
```
