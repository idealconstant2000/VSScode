fname = input('Enter file:')
if len(fname) < 1 : fname = "clown.txt"
fhand = open(fname)

many = dict()
for line in fhand:
    line = line.rstrip()
    wds = line.split()

    for w in wds:
        many[w] = many.get(w,0) + 1

# find the top 5 words by frequency

#print(many.items())
#print(sorted(many.items()))
tmp = dict()
newlist = list()


for k,v in many.items():
    tup = (v,k)
    newlist.append(tup)

#print(newlist)
cool = (sorted(newlist, reverse=True))
print(cool)

for v,k in cool[:5]:
    print(k,v)


# bigcount = None
# bigword = None
# for word,count in counts.items():
#     if bigcount is None or count > bigcount:
#         bigword = word
#         bigcount = count
# 
# print(bigword, bigcount)
