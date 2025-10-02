# x = { 'chuck' : 1 , 'fred' : 42, 'jan': 100}
# y = x.items()
# print(type(y))

# days = ('Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun')
# print(days[2])

#Which of the following methods work both in Python lists and Python tuples?
#.
# Question 8
# Using the following tuple, how would you print 'Wed'?

# Question 9
# In the following Python loop, why are there two iteration variables (k and v)?

name = input("Enter file:")
if len(name) < 1:
    name = "mbox-short.txt"
handle = open(name)

counts = dict()

for line in handle:
    words = line.split()
    # skip empty lines
    if len(words) < 2:
        continue
    # check if line starts with "From"
    if words[0] == "From":
        time = words[5]
#        print(time)
        hour = time[:2]
        counts[hour] = counts.get(hour, 0) + 1
#        print(hour)
#        counts[email] = counts.get(email, 0) + 1

#print(counts)
lst = list()
for key, val in counts.items():
    newtup = (key, val)
    lst.append(newtup)
    
lst = sorted(lst)

for val, key in lst:
    print(val, key)

