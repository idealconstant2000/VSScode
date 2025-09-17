# Dictionary with lists and nested dictionaries


store = {
    "aisle_1": ["fruits", "vegetables", "dairy"],
    "aisle_2": ["meat", "frozen", "bread"]
}
print(store["aisle_2"][1])  

complex_store = {
    "one_aisle": {
        "dairy": [{"name": "milk", "price": 2.59}, {"name": "cheese", "price": 1.59}]
    }
}
print(complex_store["one_aisle"]["dairy"][0]["price"])

students = [
    {"name": "Alice", "grade": "A"},
    {"name": "Bob", "grade": "B"},
    {"name": "Charlie", "grade": "C"}
]
print(students[1]["name"])

cart = {
    "electronics": [
        {"item": "Laptop", "price": 799.99, "quantity": 1},
        {"item": "Headphones", "price": 59.99, "quantity": 2}
    ],
    "groceries": [
        {"item": "Bananas", "price": 0.59, "quantity": 6},
        {"item": "Eggs", "price": 2.99, "quantity": 1}
    ]
}
print(cart["electronics"][0]["price"])   
print(cart["groceries"][0]["quantity"])  