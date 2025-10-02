# ITP Week 3 Day 1 Exercise

# ENUMERATE!

# 1. Read all instructions first!
# 
# Prompt: given a list of names, create a list of dictionaries with the index as the user_id and name

users_list = ["Alex", "Bob", "Charlie", "Dexter", "Edgar", "Frank", "Gary"]

# example output    
# [{"user_id": 0, "name": "Alex"}, etc, etc]

# 1a. Create a function that takes a single string value and returns the desired dictionary
def make_user_dict(name, user_id):
    return {"user_id": user_id, "name": name}

# 1b. Create a new empty list called users_dict_list
users_dict_list = []

# 1c. Loop through users_list that calls the function for each item and appends the return value to users_dict_list
for user_id, name in enumerate(users_list):
    users_dict_list.append(make_user_dict(name, user_id))

print(users_dict_list)

# 2. Prompt: Given a series of dictionaries and desired output (mock_data.py), can you provide the correct commands?
from mock_data import mock_data
results = mock_data["results"]
# 2a. retrieve the gender of Morty Smith
morty_gender = results[1]["gender"]
print(morty_gender)

# 2b. retrieve the length of the Rick Sanchez episodes
rick_episode_count = len(results[0]["episode"])
print(rick_episode_count)

# 2c. retrieve the url of Summer Smith location
summer_location_url = results[2]["location"]["url"]
print(summer_location_url)