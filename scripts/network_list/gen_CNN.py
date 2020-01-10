import itertools
import argparse
import os

MIN_DENSE = 1
MAX_DENSE = 2
ACT = 'LRELU'
KERNELS = [3, 5]
CHANNELS = [12, 32, 64, 96]
DEPTH = [32, 64, 128]
LAYERS = [['C','C','C','C'],['C','C','C'],['C','C'],['C'],['C','C','P','C','C'],['C','C','P','C','C','P'],['C','P','C'],['C','P','C','P']]
OUTPUT = '.'


dense_combinations = []
for layers in range(MIN_DENSE, MAX_DENSE+1):
    for comb in itertools.combinations_with_replacement(DEPTH, layers):
        for comb in itertools.permutations(comb):
            dense_combinations.append(comb)

dense_combinations = list(set(dense_combinations))
dense_combinations.sort()

channel_combinations = []
for layers in range(1, len(CHANNELS)+1):
    for comb in itertools.combinations_with_replacement(CHANNELS, layers):
        for comb in itertools.permutations(comb):
            channel_combinations.append(comb)

channel_combinations = list(set(channel_combinations))
channel_combinations.sort()


conv_layers = []
base = 'CNN_'+ACT

for ker in KERNELS:
    for pattern in LAYERS:
        len_pattern = len(list(filter(lambda x: x != 'P', pattern)))
        for c_comb in channel_combinations:
            if len(c_comb) == len_pattern:
                tmp = base
                idx = 0
                for layer_type in pattern:
                    if layer_type == 'C':
                        tmp = tmp+'_k'+str(ker)+'c'+str(c_comb[idx])
                        idx += 1
                    else:
                        tmp = tmp +'_p'
                conv_layers.append(tmp)

dense_layers = []
for comb in dense_combinations:
    tmp = ''
    for layer in comb:
        tmp = tmp + '_d'+str(layer)
    dense_layers.append(tmp)

networks = []
for conv in conv_layers:
    for dense in dense_layers:
        networks.append(conv+dense)

with open(os.path.join(OUTPUT,'CNN_networks_'+ACT+'.txt'), 'w') as nets:
    for net in networks:
        nets.write(net + os.linesep)

print(str(len(conv_layers))+' unique encoders')
print(str(len(dense_layers))+' unique decoders')
print(str(len(networks))+' networks generated')
