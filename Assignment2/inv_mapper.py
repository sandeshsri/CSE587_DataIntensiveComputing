#!/usr/bin/env python3
import sys
import os

punctuations = [",",".",":","[","]","#","@","!","(",")","(",")","'","’","~","$","%","^","*","&",";","/","=","-","—","_","{","}","?","\"","\“","\”","+","<",">","|","\\","°"]
stop_words = ["about","above","after","again","against","all","am","an","and","any","are","as","at"
              "be","because","been","before","being","below","between","both","but","by",
              "can","cannot","could","did","do","does","doing","down","during","each","few","for","from","further",
              "had","has","have","having","he","her","here","her","herself","him","himself","his","how",
              "if","in","into","is","it","its","itself","let","me","more","most","my","myself",
              "no","nor","not","of","off","on","once","only","or","other","ought","our","ours","ourself","out","over","own",
              "same","shall","she","should","so","some","such","than","that","the","their","theirs","them","themself","then",
              "there","these","they","this","those","through","to","too","under","until","up","upto"]

fileNameWithPath = os.getenv('mapreduce_map_input_file')
fileName = fileNameWithPath.split("/")[-1]
for line in sys.stdin:
    line = line.lower()
    for p in punctuations:
        line = line.replace(p," ")
    line = line.split()
    for word in line:
        if len(word) > 1 and not word.isnumeric() and word not in stop_words:
            print('%s\t%s' % (word, fileName))

