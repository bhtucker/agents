#! /usr/bin/env python

from sys import argv, exit
from collections import defaultdict
from random import random, randint, seed
from itertools import combinations, product
from datetime import datetime
from os.path import expanduser, join
from os import system, urandom
from ast import literal_eval as lEval
import networkx as nx
from scipy.special import binom

# function definitions
# compute the unconstrained Shapley values for nodes with attributes (colors)
def shapley(counts):
    R = counts['red']
    G = counts['green']
    B = counts['blue']
    Y = counts['none']
    N = R + G + B + Y
    rValue = 0
    gValue = 0
    for s in range(3, N):
    bLimit = min(s, B+1)
    normalizer = N * binom(N-1, s)
    for b in range(1, bLimit):
    gLimit = min(s-b, G) + 1
    rLimit = min(s-b, R) + 1
    blueChoices = binom(B, b)
    for g in range(1, gLimit):
    rValue += b * g * blueChoices * binom(G, g) * binom(Y+R-1, s-(b+g)) / normalizer
    for r in range(1, rLimit):
    gValue += b * r * blueChoices * binom(R, r) * binom(Y+G-1, s-(b+r)) / normalizer
    bValue = (R * B * G - R * rValue - G * gValue) / B
    return {'red': rValue, 'green': gValue, 'blue': bValue}

# program execution begins here
system('clear')

# read location of last file from command line and get location of setup file
if len(argv) > 1 :
    lastLoc = argv[1]
else:
    lastLoc = join(expanduser('~'), '/projects/RCAgents/.runSetup.txt')
try:
    with open(lastLoc, 'rU') as lastfp :
    setupLoc = lastfp.read().strip()
except IOError :
    setupLoc = raw_input('\nPlease enter the complete path to the setup file: ')
else:
    print '\nThe location for the setup file is: ' + setupLoc + '\n\n'
    resp = raw_input('If this is correct, enter "y"; if it is not, enter "n": ').lower()
    while resp != 'y' and resp != 'n' :
    resp = raw_input('\nPlease enter "y" or "n" to continue: ').lower()
    if resp == 'n' :
    setupLoc = raw_input('\nPlease enter the complete path to the setup file: ')

system('clear')

# read in setup file and write setup location out for next run
try:
    with open(setupLoc, 'rU') as setupfp :
    setup_tmp = [entry.strip() for entry in setupfp.readlines()]
except IOError:
    print '\nCould not open setup file:\n\n\t' + setupLoc \
    + '\n\nThe program has halted.\n\n'
    exit(1)
try:
    with open(lastLoc, 'r+') as lastfp :
    lastfp.write(setupLoc + '\n')
except IOError:
    with open(lastLoc, 'w') as lastfp :
    lastfp.write(setupLoc + '\n')

# build setup dictionary, converting numeric values and dictionaries as needed
setup = {}
for row in setup_tmp[1:] : # discard header row
    setup[row.split(';')[0]] = row.split(';')[1] # first column is key, second is value
setup['attributes'] = lEval(setup['attributes'])
setup['seed'] = lEval(setup['seed'])

# network parameters
nSize = lEval(setup['network_size']) # network size
nAttributes = len(setup['attributes']) # number of attribute types
density = lEval(setup['density']) # global density parameter

# get run time, set random generator seed, and create run ID
runtime = datetime.today().strftime('%Y-%m-%d %H:%M')
if setup['seed']:
    rSeed = setup['seed']
else:
    rSeed = abs(hash(urandom(20))/10000)
runID = 'Run_' + '{0:0>4}'.format(rSeed & 0xfff)
# This creates a "nice" run ID of the form "Run_nnnn"

# read in number of nodes and attribute distribution, and create attribute dictionary
attributes = defaultdict(dict)
for attribute in setup['attributes']:
    start = 0
    for value in setup['attributes'][attribute]:
        stop = start + setup['attributes'][attribute][value]
        attributes[attribute][value] = range(start, stop)
        start = stop
    if stop < 100:
        attributes[attribute]['none'] = range(stop, 100)
stop = 99

# create an empty graph
G = nx.Graph()

# create and distribute node attributes and record which nodes are in which attribute class
attributeSets = { attr: defaultdict(set) for attr in attributes }
attributeCounts = { attr: defaultdict(lambda: 0) for attr in attributes }
for i in range(nSize):
    nodeAttributes = {}
    for attribute in attributes:
        flip = randint(0, stop)
        [value] = [ x for x in attributes[attribute] if flip in attributes[attribute][x] ]
        attributeSets[attribute][value].add(i)
        attributeCounts[attribute][value] += 1
        nodeAttributes[attribute] = value
    G.add_node(i, attributes=nodeAttributes)

# get path to output file and set file specs for output files
outputPath = setup['output_path']
infoLoc = join(outputPath, runID + "-info.txt")
attrLoc = join(outputPath, runID + "-attributes.txt")
graphLoc = join(outputPath, runID + ".gexf")

# write out G to Gephi file, attribute partition dictionary
nx.write_gexf(G, graphLoc, encoding='utf-8', version='1.2draft')
with open(attrLoc, 'w') as attrfp:
    attrfp.write(str(attributeSets))

# compute some basic network stats
degreeC = nx.degree_centrality(G)
betC = nx.betweenness_centrality(G)
closeC = nx.closeness_centrality(G)
eigenC = nx.eigenvector_centrality(G)

# write graph info out to info file
with open(infoLoc, 'w') as infofp:
    infofp.write('Run ID:\t' + runID + '\n')
    infofp.write('Run time:\t' + runtime + '\n')
    infofp.write('Seed:\t{0:d}\n'.format(rSeed))
    infofp.write('Network size:\t' + str(nx.number_of_nodes(G)) + '\n')
    infofp.write('Graph type and parms:\t' + str(setup['gType']) + '\n')
    infofp.write('Attributes:\t' + str(setup['attributes']) + '\n\n')
    infofp.write('Actual attribute counts:\n')
    infofp.write('Attribute\tCount\tS-Value\n')
    for x in set(attributeCounts.keys()).difference({'none'}):
    infofp.write( '{0}\t{1:d}\t{2:.2f}\n'.format(x, attributeCounts[x], sValues[x]) )
    infofp.write('\nCentrality Measures:\n')
    infofp.write('Node\tDegree\tBetweeness\tCloseness\tEigenvector\n')
    for n in G.nodes():
    infofp.write('{0:d}\t{1:f}\t{2:f}\t{3:f}\t{4:f}\n'.format(n, degreeC[n], betC[n],
    closeC[n], eigenC[n]))

# all done now; send happy message to user
print '\nAll went well. Run has finished. Bye now.\n\n'
exit(0)
