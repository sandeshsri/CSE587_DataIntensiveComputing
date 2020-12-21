#!/usr/bin/env python3
import sys

join1 = {}
join2 = {}
for line in sys.stdin:
    line = line.strip()
    employee, name, salary, country, passcode = line.split('\t')

    #passcode is -1 for join1 data
    if(passcode == "-1"):
        join1[employee] = name
    else:
        join2[employee] = [salary,country,passcode]

for employee in join1.keys():
    name = join1[employee]
    if(employee in join2) :
        salary = join2[employee][0]
        country = join2[employee][1]
        passcode = join2[employee][2]
    else:
        salary = -1
        country = -1
        passcode = -1
    print('%s\t%s\t%s\t%s\t%s' % (employee, name, salary, country, passcode))
        
for employee in join2.keys():
    if(employee not in join1) :
        name = join2[employee]
        salary = -1
        country = -1
        passcode = -1
        print('%s\t%s\t%s\t%s\t%s' % (employee, name, salary, country, passcode))
