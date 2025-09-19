import json

example_dict = {
    "name": "John",
    "age": 30,
    "married": True,
    "divorced": False,
    "children": ["Ann", "Billy"],
    "cars": [
        {"model": "BMW 230", "mpg": 27.5},
        {"model": "Ford Edge", "mpg": 24.1}
    ]
}

# Convert dictionary to JSON
#print(json.dumps(example_dict))

#store JSON as string
# json_dict = json.dumps(example_dict)
# print(json_dict)

# Convert dictionary to indented JSON (pretty printing)
# indented_json = json.dumps(example_dict, indent=4)
# print(indented_json)

# separated_json = json.dumps(example_dict, indent=4, separators=(". ", "= "))
# print(separated_json)

# sorted_json = json.dumps(example_dict, indent=4, sort_keys=True)
# print(sorted_json)

print(json.dumps(example_dict, indent=2, separators=("; ", " = "), sort_keys=True))

