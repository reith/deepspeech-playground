import argparse
import json
import sys


def main(input_f, dict_f, out_f):
    desc_i = open(input_f)
    dic = open(dict_f)

    dic_read_line = None
    prev_desc = None
    dic_file_ended = False

    arpabet_descs = []

    for desc_line in desc_i:
        desc_ld = json.loads(desc_line)
        text = desc_ld['text'].lower()
        if dic_read_line:
            if dic_read_line[0].lower() == text:
                desc_ld['arpabet'] = dic_read_line[1]
            dic_read_line = None

        if 'arpabet' not in desc_ld:
            pronun_found = False
            while not pronun_found:
                line = dic.readline()
                if line == '':
                    sys.stderr.write('WARNING: dictionary file ended while'
                                     ' still looking for: {}\n'.format(text))
                    dic_file_ended = True
                    break
                dic_read_line = line[:-1].split('\t')
                if dic_read_line[0].lower() == text:
                    desc_ld['arpabet'] = dic_read_line[1]
                    dic_read_line = None
                    pronun_found = True
                elif prev_desc and (prev_desc['text'] ==
                                    dic_read_line[0].lower().split('(')[0]):
                    sys.stderr.write('INFO: found another pronunciation for: '
                                     '{}\n'.format(prev_desc['text']))
                    prev_desc_new = prev_desc.copy()
                    prev_desc_new['arpabet'] = dic_read_line[1]
                    arpabet_descs.append(prev_desc_new)
                else:
                    break

        if 'arpabet' in desc_ld:
            arpabet_descs.append(desc_ld)
        else:
            sys.stderr.write("WARNING: couldn't find pronunciation for: {}\n"
                             .format(text))
        prev_desc = desc_ld
        if dic_file_ended:
            sys.stderr.write('WARNING: dictionary find ended sooner\n')
            break

    with open(out_f, 'w') as out:
        for desc in arpabet_descs:
            out.write(json.dumps(desc) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_desc', type=str, help='Input json line file')
    parser.add_argument('dict', type=str,
                        help='Arpabet translation file of input desc json')
    parser.add_argument('output_desc', type=str,
                        help='Output json line file')
    args = parser.parse_args()

    main(args.input_desc, args.dict, args.output_desc)
