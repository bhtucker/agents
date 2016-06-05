from sys import argv, exit
from random import randint
from datetime import datetime
from os.path import expanduser, join
from os import urandom
from ast import literal_eval as lEval
import networkx as nx

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

# network parameters
attributes = lEval(setup['attributes'])
pEdges = lEval(setup['edge_probs'])
runSeed = lEval(setup['seed'])
nSize = lEval(setup['network_size']) # network size
density = lEval(setup['density']) # global density parameter

# get run time, set random generator seed, and create run ID
runtime = datetime.today().strftime('%Y-%m-%d %H:%M')
if !runSeed:
    runSeed = abs(hash(urandom(20))/10000)
runID = 'Run_' + '{0:0>4}'.format(runSeed & 0xfff)
# This creates a "nice" run ID of the form "Run_nnnn"

# instantiate attribute generator to distribute attributes over nodes
# takes attributes dictionary and the network size as parms
aGen = AttributeGenerator(attributes, nSize)

# instantiate edge generator to determine dyadic ties
# takes attributes dictionary,the unscaled probabilities of ties
# between nodes of similar or disimilar type, network size and density as parms
eGen = EdgeGenerator(attributes, pEdges, nSize, density)

# create an empty graph
G = nx.Graph()

# create and distribute node attributes and record which nodes are in which attribute class
attributeSets={attr: {value: set() for value in values.keys()} for attr,values in attributes.items()}
attributeCounts={attr: {value: 0 for value in values.keys()} for attr,values in attributes.items()}
for i in range(nSize):
    nodeAttributes = {}
    for attribute in attributes:
        value = aGen.get_value(attribute)
        attributeSets[attribute][value].add(i+1)
        attributeCounts[attribute][value] += 1
        nodeAttributes[attribute] = value
    G.add_node(i+1, nodeAttributes)

# iterate over dyads of nodes and set an edge between them if set_edge returns true
for dyad in combinations(nx.nodes(G), 2):
    if eGen.set_edge(G.node[dyad[0]], G.node[dyad[1]]):
            G.add_edge(dyad)

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
    infofp.write('Seed:\t{0:d}\n'.format(runSeed))
    infofp.write('Network size:\t' + str(nSize) + '\n')
    infofp.write('Network density:\t' + str(nx.density(G)) + '\n')
    infofp.write('Attributes:\t' + str(attributes) + '\n')
    infofp.write('\nActual attribute counts:\n')
    for attribute in attributeCounts:
        infofp.write('\n' + attribute + '\n')
        for value in attributeCounts[attribute]:
            infofp.write( '{0}:\t{1:d}\n'.format(value, attributeCounts[attribute][value]) )
    infofp.write('\nAttribute sets:\n')
    for attribute in attributeSets:
        infofp.write('\n' + attribute + '\n')
        for value in attributeSets[attribute]:
            infofp.write( '{0}:\t{1}\n'.format(value, str(attributeSets[attribute][value])) )
    infofp.write('\nCentrality Measures:\n')
    infofp.write('Node\tDegree\tBetweeness\tCloseness\tEigenvector\n')
    for n in G.nodes():
        infofp.write('{0:d}\t{1:f}\t{2:f}\t{3:f}\t{4:f}\n'.format(n, degreeC[n], betC[n],
            closeC[n], eigenC[n]))
