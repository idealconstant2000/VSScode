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
        email = words[1]
        counts[email] = counts.get(email, 0) + 1

bigcount = None
bigword = None
for email,count in counts.items():
    if bigcount is None or count > bigcount:
        bigemail = email
        bigcount = count
        
print(bigemail, bigcount)


