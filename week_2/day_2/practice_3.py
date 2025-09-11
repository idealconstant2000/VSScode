# ITP Week 2 Day 2 (In-Class) Practice 3

# Functions Parameters and Arguments

# Lets take those functions we built in practice_2 and make them more dynamic:

# Rewrite the functions from practice_2 using parameters:
# add_num

# subtract_num

# multiply_num

# divide_num

# Create a function called add_num: 

def add_num(x, y):
    # x = 5
    # y = 7
    print(x + y)

# Inside your function define two variables: x and y, assign 5 to x and 7 to y
# print the sum of x and y

# Call your function
# add_num()


# Create a function called subtract_num:

def subtract_num(x, y):

# Inside your function define two variables: x and y, assign 10 to x and 3 to y
# print the difference of x and y

    # x = 10
    # y = 3
    print(x - y)

# Call your function

# subtract_num()

# Create a function called multiply_num:

def multiply_num(x, y):

# Inside your function define two variables: x and y, assign 5 to x and 7 to y
# print the product of x and y
    # x = 5
    # y = 7
    print(x * y)

# Call your function

# multiply_num()


# Create a function called divide_num:

def divide_num(x, y):

# Inside your function define two variables: x and y, assign 10 to x and 2 to y
# print the quotient of x divided by y

    # x = 10
    # y = 2
    print(x / y)

# Don't forget to call your functions to make sure they work

#Uncomment to call your functions:
# print("I should see the number 7 below from add_num: ")
# add_num(3, 4)

# print("I should see the number -2 below from subtract_num: ")
# subtract_num(6, 8)

# print("I should see the number 18 below from multiply_num: ")
# multiply_num(2, 9)

# print("I should see the number 2 below from divide_num: ")
# divide_num(10, 5)

# Extra Time?

# Now take in 2 users inputs and pass them 
# in as arguments when calling the functions

x = int(input("Please enter your first number: "))
y = int(input("Please enter your second number: "))
print("I should see the sum of your two numbers below from add_num: ")
add_num(x, y)
print("I should see the difference of your two numbers below from subtract_num: ")
subtract_num(x, y)
print("I should see the product of your two numbers below from multiply_num: ")
multiply_num(x, y)
print("I should see the quotient of your two numbers below from divide_num: ")
divide_num(x, y)   

