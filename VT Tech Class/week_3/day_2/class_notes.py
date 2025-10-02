# Example of converting JSON text into a Python dictionary
#import json
# from dummy_json import dummy_json   # pretend file with JSON text

# # Convert JSON string into Python dictionary
# converted_json_dictionary = json.loads(dummy_json)

# #print(converted_json_dictionary)
# print(converted_json_dictionary['researcher']['relatives'][0]['name'])

# converted_json = json.dumps({"test": "good"})
# print(converted_json) 

import requests
url = 'https://jsonplaceholder.typicode.com/posts'
response = requests.get(url)

print(response)          # Shows the Response object


json_data = response.json()  # Converts response directly into Python objects
#print(json_data)          # Shows the data structure
print(type(json_data))    # Shows the type of the data structure
print(len(json_data))     # Shows the length of the data structure

# json_data is now a list of dictionaries
for post in json_data[:5]:
    print(post['title']) 