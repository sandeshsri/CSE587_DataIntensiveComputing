#!/usr/bin/env python3
import sys

# input comes from STDIN (standard input)
for line in sys.stdin:
    line = line.strip()
    line = line.split("\t")
    employee = "-1"
    name = "-1"
    salary = "-1"
    country = "-1"
    passcode = "-1"

    #employee ID is common
    employee = line[0]

    #Ignore headers
    if(employee == "Employee ID"):
        continue

    #If length is 4, it is join 2 data
    if len(line) ==4:
        salary = line[1]
        country = line[2]
        passcode = line[3]
    # length is 2, is is join1 data
    else:
         name = line[1]

    print('%s\t%s\t%s\t%s\t%s' % (employee, name, salary, country, passcode))
