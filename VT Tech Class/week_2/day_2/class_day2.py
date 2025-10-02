# from openpyxl import load_workbook 
# wb = load_workbook("./week_2/spreadsheets/wk2_day_1_practice.xlsx")

# # print(wb.sheetnames)

# ws = wb['Sheet'] 
# all_rows = ws.rows

# print(list(all_rows))

# Count rows

# count = 0
# for row in ws.values:
#     count += 1
# print("total rows: ", count)

# update data
# ws['A1'] = "Hello from Day 2!"
# ws.cell(row = 2, column = 2).value = "Day 2 update"
# wb.save("./week_2/spreadsheets/wk2_day_2.xlsx") 

# creating a function
# def greetings():
#     print("Hello World!")

# greetings()

# def add_five(num):
# #    print(num + 5)
#     return num + 5
# result = add_five(10)
# #print("Result is:", result)  # None

# print(f"Result is: {result}") 


# multiplep arameters

def add_five(any_number):
    print(any_number + 5)
add_five(10)
add_five(20)

def multi_parameter(first_name, last_name, location):
    print("My name is " + first_name + " " + last_name + ", and I live in " + location + ".")

multi_parameter("Daniel", "Kim", "Los Angeles")