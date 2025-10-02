# ITP Week 3 Day 2 Exercise

# import in the two modules for making API calls and parsing the data

import requests
url = "https://rickandmortyapi.com/api/character"
response = requests.get(url)

# set a url variable to "https://rickandmortyapi.com/api/character"

# set a variable response to the "get" request of the url

# print to verify we have a status code of 200
print(response)

# assign a variable json_data to the responses' json

import json
#json_data = response.json()
json_data = response.text

# print to verify a crazy body of strings!
print(json_data)

# lets make it into a python dictionary by using the appropriate json method
python_data = json.loads(json_data)


# print the newly created python object
print(type(python_data))
print(python_data)
