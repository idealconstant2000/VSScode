fh = open("mbox-short.txt")

for line in fh:
    ly = line.rstrip()
    print(ly.upper())