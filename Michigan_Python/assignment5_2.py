largest = None
smallest = None
while True:
    num = input("Enter a number: ")
    if num == "done":
        break
    try:
        num = float(num)
    except:
        print("Invalid input")
        continue 

    if largest is None or num > largest: 
              largest = num
    if smallest is None or num < smallest:
              smallest = num
              
#   print(num)

print("Maximum is ", largest)
print("Minimum is ", smallest)
