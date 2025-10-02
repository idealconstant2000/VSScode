# Use the file name mbox-short.txt as the file name
filename = input("Enter file name: ")
fh = open(filename)
linecount = 0 
pos = 0
sval = 0
fval = 0
tval = 0

for line in fh:
    if line.startswith("X-DSPAM-Confidence:"):
            pos = line.find(":")
            sval = line[pos+2:]
            fval = float(sval.strip())
            tval = tval + fval
            linecount = linecount + 1
    else:
        continue
#print(pos)
#print(sval)
#print(tval)
#print(linecount)
result = tval/linecount
print("Average spam confidence:",result)

