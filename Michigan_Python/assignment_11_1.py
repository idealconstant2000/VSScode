import re
fname = input("Enter file name")
if len(fname) < 1:
    fname = "sample.txt"

fh = open(fname)
total = 0

for line in fh:
    line = line.strip()
    numbers = re.findall('[0-9]+', line)
    for num in numbers:
        total += int(num)
        
print(total)
    
    




