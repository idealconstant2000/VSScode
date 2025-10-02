fname = input("Enter file name: ")
if len(fname) < 1:
    fname = "mbox-short.txt"

fh = open(fname)
email = []
count = 0
words = []

for line in fh:
    words = line.split()
    if len(words) < 2:
        continue
    if words[0] == "From":
        email.append(words[1])
#        count += 1
count = len(email)
for i in email:
    print(i)

print("There were", count, "lines in the file with From as the first word")
