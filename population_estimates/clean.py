# Create a dictionary out of the UN World Pop estimates
import csv
import pprint

with open('wpp.csv', mode='r') as infile:
    reader = csv.reader(infile)
    _cleaner = lambda x: float(x)/1000
    mydict = {rows[0]:map(_cleaner, rows[1:]) for rows in reader}

pprint.pprint(mydict)