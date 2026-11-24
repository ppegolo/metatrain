"""Generate the PySCF references of ``tests/utils/test_density.py``.

    python generate.py

Writes ``basis.json``, a small basis in the JSON format of the Basis Set Exchange with
contracted shells, several shells per angular momentum and angular momenta up to 6,
and ``reference.json``, with for each system the overlap and Coulomb matrices
multiplied by the vector ``sin(1), sin(2), ...`` and the integral of each basis
function. Run ``tox -e format`` afterwards to indent the files as the repository does.
"""

import json

import numpy as np
from pyscf import gto
from pyscf.data.elements import ELEMENTS
from pyscf.gto.ft_ao import ft_ao


# (angular momentum, [(exponent, coefficient), ...]) for each atomic number
BASIS = {
    1: [
        (0, [(15.6752927, 0.0186886), (3.6063578, 0.0631670), (1.2080016, 0.1204609)]),
        (0, [(0.4726794, 1.0)]),
        (1, [(2.0281365, 0.0537148), (0.5358730, 0.0480413)]),
    ],
    8: [
        (0, [(2876.8216605, 0.2087695), (585.1394347, 0.2896498), (164.2365803, 0.7)]),
        (0, [(16.9441120, 1.0)]),
        (1, [(31.2549750, 1.0)]),
        (1, [(4.6740910, 0.6), (1.4210330, 0.4)]),
        (2, [(3.2620470, 1.0)]),
    ],
    64: [
        (0, [(64.3, 0.4), (21.2, 0.6)]),
        (0, [(3.1, 1.0)]),
        (1, [(9.8, 1.0)]),
        (2, [(12.4, 0.5), (4.3, 0.5)]),
        (2, [(1.6, 1.0)]),
        (3, [(5.7, 1.0)]),
        (4, [(4.1, 1.0)]),
        (5, [(2.9, 1.0)]),
        (6, [(2.2, 1.0)]),
    ],
}

SYSTEMS = [
    ([8, 1, 1], [[0.0, 0.0, 0.0], [0.0, 0.76, 0.59], [0.0, -0.76, 0.59]]),
    (
        [64, 8, 1, 1],
        [[0.0, 0.0, 0.0], [2.4, 0.1, 0.0], [2.9, 0.9, 0.2], [2.8, -0.7, -0.3]],
    ),
]

basis = {
    "molssi_bse_schema": {"schema_type": "minimal", "schema_version": "0.1"},
    "name": "test",
    "description": "small basis for the tests of the density loss",
    "elements": {
        str(z): {
            "electron_shells": [
                {
                    "function_type": "gto",
                    "region": "",
                    "angular_momentum": [angular],
                    "exponents": [str(e) for e, _ in primitives],
                    "coefficients": [[str(c) for _, c in primitives]],
                }
                for angular, primitives in shells
            ]
        }
        for z, shells in BASIS.items()
    },
}
with open("basis.json", "w") as fd:
    json.dump(basis, fd, indent=1)

pyscf_basis = {
    ELEMENTS[z]: [[angular, *map(list, primitives)] for angular, primitives in shells]
    for z, shells in BASIS.items()
}
references = []
for types, positions in SYSTEMS:
    atoms = [(ELEMENTS[z], p) for z, p in zip(types, positions, strict=True)]
    mol = gto.M(atom=atoms, basis=pyscf_basis, spin=None, unit="Angstrom")
    vector = np.sin(np.arange(1, mol.nao + 1))
    references.append(
        {
            "types": types,
            "positions": positions,
            "overlap": (mol.intor("int1e_ovlp") @ vector).tolist(),
            "coulomb": (mol.intor("int2c2e") @ vector).tolist(),
            "integrals": ft_ao(mol, np.zeros((1, 3)))[0].real.tolist(),
        }
    )
with open("reference.json", "w") as fd:
    json.dump(references, fd)
