import re
# 
hand = open("mbox-short.txt")
# 
# for line in hand:
#     line = line.rstrip()
#     if re.search('^From:', line):
#         print(line)

x = "My 2 favorite numbers are 19 and 42."
# y = re.findall('[0-9]+', x)
# print(y)

# y = re.findall('[AEIOU]+',x)
# print(y)

# for line in hand:
#     if line.startswith("From "):
#         line = line.rstrip()
#         y = re.findall('^From .*@([^ ]*)',line)
#         print(y)

numlist = list()
for line in hand:
    line = line.rstrip()
    stuff = re.findall('X-DSPAM-Confidence: ([0-9.]+)', line)
    if len(stuff) != 1 : continue
    num = float(stuff[0])
    numlist.append(num)
print("Maximum: ", max(numlist))
    
    
                