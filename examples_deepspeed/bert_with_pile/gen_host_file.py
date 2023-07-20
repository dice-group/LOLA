import os
env_node_list = os.environ['SLURM_NODELIST']
gpus_per_node = os.environ['SLURM_GPUS_ON_NODE']
# ref https://gist.github.com/ebirn/cf52876120648d7d85501fcbf185ff07
def parse_int(s):
    for i,c in enumerate(s):
        if c not in "0123456789":
            return int(s[:i]), s[i:]
    return int(s), ""

def parse_brackets(s):
    # parse a "bracket" expression (including closing ']')
    lst = []
    while len(s) > 0:
        if s[0] == ',':
            s = s[1:]
            continue
        if s[0] == ']':
            return lst, s[1:]
        a, s = parse_int(s)
        assert len(s) > 0, f"Missing closing ']'"
        if s[0] in ',]':
            lst.append(a)
        elif s[0] == '-':
            b, s = parse_int(s[1:])
            lst.extend(range(a,b+1))
    assert len(s) > 0, f"Missing closing ']'"

def parse_node(s):
    # parse a "node" expression
    for i,c in enumerate(s):
        if c == ',': # name,...
            return [ s[:i] ], s[i+1:]
        if c == '[': # name[v],...
            b, rest = parse_brackets(s[i+1:])
            if len(rest) > 0:
                assert rest[0] == ',', f"Expected comma after brackets in {s[i:]}"
                rest = rest[1:]
            return [s[:i]+str(z) for z in b], rest

    return [ s ], ""

def parse_list(s):
    lst = []
    while len(s) > 0:
        v, s = parse_node(s)
        lst.extend(v)
    return lst

nodes = parse_list(env_node_list)

with open('hostfile','w') as hostfile:
    for item in nodes:
        # check if its a GPU node
        if 'n2gpu' in item:
            hostfile.write(item+' slots='+gpus_per_node+'\n')
