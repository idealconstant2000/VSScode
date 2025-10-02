fname = input("Enter file name: ")
fh = open(fname)

words = []

lst = list()
for line in fh:
    for word in line.split():
        if word not in words:
            words.append(word)
words.sort()
print(words)