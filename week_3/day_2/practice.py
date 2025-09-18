# ITP Week 3 Day 2 Practice

# 1. import the appropriate module
import json

json_1 = """
{
    "albumId": 1,
    "id": 1,
    "title": "accusamus beatae ad facilis cum similique qui sunt",
    "url": "https://via.placeholder.com/600/92c952",
    "thumbnailUrl": "https://via.placeholder.com/150/92c952"
}
"""

# 2. perform a deserialization of the above object
deserial_json_1 = json.loads(json_1)
print(deserial_json_1)

# 3. assign a new variable called url_1 to the value of the deserialized object's url

url_1 = deserial_json_1["url"]
print("url_1:", url_1)


json_2="""
[
{
"albumId": 1,
"id": 1,
"title": "accusamus beatae ad facilis cum similique qui sunt",
"url": "https://via.placeholder.com/600/92c952",
"thumbnailUrl": "https://via.placeholder.com/150/92c952"
},
{
"albumId": 1,
"id": 2,
"title": "reprehenderit est deserunt velit ipsam",
"url": "https://via.placeholder.com/600/771796",
"thumbnailUrl": "https://via.placeholder.com/150/771796"
}
]
"""

# 4. deserialize and assign a variable url_2 with the second item's url
deserial_json_2 = json.loads(json_2)
print(deserial_json_2)
url_2 = deserial_json_2[1]["url"]
print("url_2:", url_2)
#print(type(deserial_json_2))


