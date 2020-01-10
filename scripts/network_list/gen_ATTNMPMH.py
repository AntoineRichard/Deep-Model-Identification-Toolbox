import itertools
import argparse
import os

MIN_DENSE = 1
MAX_DENSE = 3
MIN_ALPHA = 1
MAX_ALPHA = 5
ACT = 'LRELU'
MODEL_DEPTH = [16, 32, 48, 64, 128]
DEPTH = [16, 32, 64, 96, 128]
OUTPUT = '.'

combinations = []
for layers in range(MIN_DENSE, MAX_DENSE+1):
    for comb in itertools.combinations_with_replacement(DEPTH, layers):
        for comb in itertools.permutations(comb):
            combinations.append(comb)

combinations = list(set(combinations))
combinations.sort()

networks = []
base = 'ATTNMPMH_'+ACT
for ALPHA in range(MIN_ALPHA, MAX_ALPHA+1):
    tmp = base + '_a'+str(ALPHA)
    for MD in MODEL_DEPTH:
        tmp2 = tmp + '_md'+str(MD)
        for comb in combinations:
            tmp3 = tmp2
            for layer in comb:
                tmp3 = tmp3 + '_d'+str(layer)
            networks.append(tmp3)

with open(os.path.join(OUTPUT,'ATTNMPMH_networks_'+ACT+'.txt'), 'w') as nets:
    for net in networks:
        nets.write(net + os.linesep)

print(str(len(networks))+' networks generated')
