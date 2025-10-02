counts = dict()
names = ['csev', 'cwen', 'csev', 'zqian', 'cwen']

for name in names:
    if name not in counts:
        counts[name] = 1
    else:
        counts[name] = counts[name] +1
print(counts)

counter = dict()
names_2 = ['csev', 'cwen', 'csev', 'zqian', 'cwen']
for name_2 in names_2:    
    counter[name_2] = counter.get(name_2, 0) + 1
print(counter)

