from utils import for_tf_or_th

sphinx_40_phones = '''
AA
AE
AH
AO
AW
AY
B
CH
D
DH
EH
ER
EY
F
G
HH
IH
IY
JH
K
L
M
N
NG
OW
OY
P
R
S
SH
T
TH
UH
UW
V
W
Y
Z
ZH
IX
'''
# SIL was not exists MakeDict output but there was IX translated for '
# sphinx_40_phones = sphinx_40_phones[:-4] + 'IX'

phone_map, index_map = {}, {}
start_index = for_tf_or_th(0, 1)
for i, ph in enumerate(sphinx_40_phones.strip().split()):
    phone_map[ph] = i + start_index
    index_map[i+start_index] = ph


def arpabet_to_int_sequence(phonograms):
    return [phone_map[ph] for ph in phonograms.split()]
