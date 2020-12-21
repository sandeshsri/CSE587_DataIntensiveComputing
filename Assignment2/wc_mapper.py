#!/usr/bin/env python3
import sys

punctuations = [",",".",":","[","]","#","@","!","(",")","(",")","'","’","~","$","%","^","*","&",";","/","=","-","—","_","{","}","?","\"","\“","\”","+","<",">","|","\\","°"]
stop_words = ["about","above","after","again","against","all","am","an","and","any","are","as","at"
              "be","because","been","before","being","below","between","both","but","by",
              "can","cannot","could","did","do","does","doing","down","during","each","few","for","from","further",
              "had","has","have","having","he","her","here","her","herself","him","himself","his","how",
              "if","in","into","is","it","its","itself","let","me","more","most","my","myself",
              "no","nor","not","of","off","on","once","only","or","other","ought","our","ours","ourself","out","over","own",
              "same","shall","she","should","so","some","such","than","that","the","their","theirs","them","themself","then",
              "there","these","they","this","those","through","to","too","under","until","up","upto"]

for line in sys.stdin:
    line = line.lower()
    for p in punctuations:
        line = line.replace(p," ")
    line = line.split()
    for word in line:
        employee = "-1"
        name = "-1"
        pref = "-1"
        pref_name = "-1"
        good = "-1"
        if len(word) > 1 and not word.isnumeric() and word not in stop_words:
            print('%s\t%s' % (word, 1))
            


#!/usr/bin/python

import sys

# input comes from STDIN (standard input)
for line in sys.stdin:
    line = line.strip()
    line = line.split(",")

    


    #If length of the line is four, the data comes from customerData
    if len(line) ==4:
        id = line[0]
        name = line[1]
        pref = line[2]
        good = line[3]

    else:
        pref = line[0]
        pref_name = line[1]


    print '%s\t%s\t%s\t%s\t%s' % (id, name, pref, pref_name, good)