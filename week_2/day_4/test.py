# colors = {'red', 'blue', 'red', 'green', 'blue'}
# print(len(colors))


# my_car = {
#   "brand": "Ford",
#   "model": "Mustang",
#   "year": 2021,
#   "interest": 22
# }

# #print("My Car Dictionary:", my_car)

# print(my_car["year"]) 

# car_year = my_car.get("year")
# print(car_year)

# your_car = {
#   "brand": "Toyota",
#   "model": "Prius",
#   "electric": True,
#   "year": 2012,
#   "colors": ["red", "white", "blue"]
# }
# print(your_car.keys())  
# print(your_car.values())  
# print(your_car.items())

# car = {"brand": "Ford", "year": 1964}
# print("Original:", car)

# car["color"] = "red"   # Add a new key:value
# print("After adding color:", car)

# car.update({"year": 2020})  # Update existing or add new
# print("Updated:", car)

# car["brand"] = "Chevy"
# print("Updated brand:", car)


# if "model" in car:
#     print("'model' key exists")
# else:
#     print("'model' not found")

# print(len(car))

# car = {
#   "brand": "Ford",
#   "model": "Mustang",
#   "year": 1964
# }
# print("Original:", car)

# car.pop("year")      # Remove by key
# print("After pop:", car)

# car.popitem()        # Remove last inserted
# print("After popitem:", car)

# car = {
#   "brand": "Ford",
#   "model": "Mustang",
#   "year": 1964
# }

# del car["brand"]     # Delete specific key
# print("After del:", car)

# car.clear()          # Empty dictionary
# print("After clear:", car)

car = {
  "brand": "Ford",
  "model": "Mustang",
  "year": 2021
}

for value in car.values():
    print(f"Value: {value}")

for key in car.keys():
    print(f"Key: {key}")

for key, value in car.items():
    print(f"Print key value: {key, value}")



