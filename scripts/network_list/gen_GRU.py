import itertools
import argparse
import os

MIN_DENSE = 1
MAX_DENSE = 3
MIN_HIDDEN_LAYERS = 1
MAX_HIDDEN_LAYERS = 3
ACT = 'LRELU'
HIDDEN_STATE_SIZE = [16, 32, 48, 64, 96, 128]
DEPTH = [16, 32, 48, 64, 96,128]
OUTPUT = '.'

combinations = []
for layers in range(MIN_DENSE, MAX_DENSE+1):
    for comb in itertools.combinations_with_replacement(DEPTH, layers):
        for comb in itertools.permutations(comb):
            combinations.append(comb)

combinations = list(set(combinations))
combinations.sort()

networks = []
base = 'GRU_'+ACT
for HSS in HIDDEN_STATE_SIZE:
    tmp = base + '_hs'+str(HSS)
    for HL in range(MIN_HIDDEN_LAYERS, MAX_HIDDEN_LAYERS+1):
        tmp2 = tmp + '_l'+str(HL)
        for comb in combinations:
            tmp3 = tmp2
            for layer in comb:
                tmp3 = tmp3 + '_d'+str(layer)
            networks.append(tmp3)

with open(os.path.join(OUTPUT,'GRU_networks_'+ACT+'.txt'), 'w') as nets:
    for net in networks:
        nets.write(net + os.linesep)

print(str(len(networks))+' networks generated')
