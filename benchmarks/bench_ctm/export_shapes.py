import os
import json
# import yastn

def export_shapes(a, Tt, Tr, fheader=None, a_order="tlbrsa", Tt_order="lacr", Tr_order="tacb"):
    """
    Read legs of yastn tensors a, Tt, Tr;
    Generate dictionary compatible with data in /input_shapes
    """
    a_legs = unfuse_legs(a.get_legs())
    if len(a_legs) < len(a_order):
        a_order = "".join(x for x in a_order if x != 'a')
    a_legs = dict(zip(a_order, a_legs))
    Tt_legs = dict(zip(Tt_order, unfuse_legs(Tt.get_legs())))
    Tr_legs = dict(zip(Tr_order, unfuse_legs(Tr.get_legs())))

    d = {"symmetry": a.config.sym.SYM_ID}
    for dirn, leg in a_legs.items():
        d[f"a_leg_{dirn}"] = dict_leg(leg)

    for dirn in "lr":
        d[f"Tt_leg_{dirn}"] = dict_leg(Tt_legs[dirn])

    for dirn in "tb":
        d[f"Tr_leg_{dirn}"] = dict_leg(Tr_legs[dirn])

    fname = f"{a.config.sym.SYM_ID}_d={sum(a_legs['s'].D)}"
    if 'a' in a_legs:
        fname = fname + f"x{sum(a_legs['a'].D)}"
    D = max(sum(a_legs[dirn].D) for dirn in 'tlbr')
    chi = max(*(sum(Tt_legs[dirn].D) for dirn in 'lr'),
              *(sum(Tr_legs[dirn].D) for dirn in 'tb'))
    fname = fname + f"_{D=}_{chi=}.json"
    if fheader is not None:
        fname = f"{fheader}_" + fname
    fname = os.path.join(os.path.dirname(__file__), "input_shapes", fname)

    txt = ["{\n"]
    for k, v in d.items():
        txt.append(f'    "{k}": {json.dumps(v)}')
        txt.append(',\n')
    txt.pop()
    txt.append("\n}")
    txt = "".join(txt)
    with open(fname, 'w', encoding='utf-8') as f:
        f.write(txt)


def dict_leg(leg):
    d = {"signature": leg.s,
         "charges": [list(x) for x in leg.t],
         "dimensions": list(leg.D)}
    return d


def unfuse_legs(legs):
    while any(leg.is_fused() for leg in legs):
        tmp = []
        for leg in legs:
            if leg.is_fused():
                for x in leg.unfuse_leg():
                    tmp.append(x)
            else:
                tmp.append(leg)
        legs = tmp
    return legs
