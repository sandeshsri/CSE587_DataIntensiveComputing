import sys
import operator

wordcount = {}
for line in sys.stdin:
    line = line.strip()
    word, count = line.split('\t', 1)
    try:
        count = int(count)
    except ValueError:
        continue

    try:
        wordcount[word] = wordcount[word] + count
    except:
        wordcount[word] = count

top_words = dict(sorted(wordcount.items(), key = operator.itemgetter(1), reverse=True))
count = 0
for word in top_words.keys():
    count += 1
    print('%s\t%s' % (word, top_words[word]))
    if count == 10:
        break