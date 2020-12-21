import sys

punctuations = [",",".",":","[","]","#","@","!","(",")","(",")","'","’","~","$","%","^","*","&",";","/","=","-","—","_","{","}","?","\"","\“","\”","+","<",">","|","\\","°"]
keys = ["science", "sea", "fire"]
temp = None
previous = None
current = None
next = None

for line in sys.stdin:
    line = line.lower()
    for p in punctuations:
        line = line.replace(p," ")
    line = line.split()
    if(not len(line) > 0):
        continue

    for i in range(len(line)):
        current = line[i]
        if temp:
            if previous == keys[0] or previous == keys[1] or previous == keys[2]:
                word = temp + "_$_" + current
                print('%s\t%s' % (word, 1))
            temp = None
        if i+1 < len(line):
            next = line[i+1]
            if previous and next and current == keys[0] or current == keys[1] or current == keys[2]:
                word = previous + "_$_" + next
                print('%s\t%s' % (word, 1))
            previous = line[i]
        else:
            temp = previous
            previous = line[i]

