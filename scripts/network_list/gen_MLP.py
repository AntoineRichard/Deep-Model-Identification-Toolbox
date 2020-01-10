import itertools
import argparse
import os

MIN_LAYERS = 1
MAX_LAYERS = 4
ACT = 'LRELU'
DEPTH = [16, 32, 48, 64, 96, 128, 256]
OUTPUT = '.'

combinations = []
for layers in range(MIN_LAYERS, MAX_LAYERS+1):
    for comb in itertools.combinations_with_replacement(DEPTH, layers):
        for comb in itertools.permutations(comb):
            combinations.append(comb)

combinations = list(set(combinations))
combinations.sort()

networks = []
base = 'MLP_'+ACT
for comb in combinations:
    tmp = base
    for layer in comb:
        tmp = tmp + '_d'+str(layer)
    networks.append(tmp)

with open(os.path.join(OUTPUT,'MLP_networks_'+ACT+'.txt'), 'w') as nets:
    for net in networks:
        nets.write(net + os.linesep)

print(str(len(networks))+' networks generated')
