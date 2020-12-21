#!/usr/bin/env python3
import sys

files = {}
for line in sys.stdin:
    line = line.strip()
    word, doc = line.split('\t', 1)
    continue

    try:
        files[word].add(doc)
    except:
        files[word] = {doc}

for word in files.keys():
    print('%s\t%s' % (word, files[word]))
