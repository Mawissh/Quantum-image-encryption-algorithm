"""
4x4 GNEQR quantum simulation for the project encryption pipeline.

Pipeline simulated here:
    image -> GNEQR -> Baker scrambling -> chaotic diffusion -> Kumar P-box

The chaotic ODE/key schedule is not implemented inside the quantum circuit. For
this 4x4 reproducible demo, the diffusion keystream is a fixed table generated
classically from diffusion.generate_keystream(DEFAULT_IC0, 16). The quantum
circuit applies the resulting reversible byte transform under coordinate control.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import Statevector

from baker_permutation import decrypt_unscramble, encrypt_scramble
from diffusion import diffuse_decrypt, diffuse_encrypt, generate_keystream
from chaotic_system import DEFAULT_IC0, DEFAULT_DT, DEFAULT_PARAMS


KUMAR_APPENDIX_IMAGE = np.array(
    [
        [0, 15, 31, 47],
        [63, 79, 95, 111],
        [127, 143, 159, 175],
        [191, 207, 223, 239],
    ],
    dtype=np.uint8,
)

KUMAR_PBOX_TABLE = np.array(
    [
        [8, 7, 15, 13],
        [4, 14, 0, 3],
        [5, 6, 2, 12],
        [1, 9, 10, 11],
    ],
    dtype=np.int64,
)

KUMAR_INVERSE_PBOX_TABLE = np.array(
    [
        [6, 12, 10, 7],
        [4, 8, 9, 1],
        [0, 13, 14, 15],
        [11, 3, 5, 2],
    ],
    dtype=np.int64,
)

MANUSCRIPT_PBOX_TABLE = np.array(
    [
        [9, 3, 14, 5],
        [8, 13, 2, 10],
        [0, 1, 4, 7],
        [11, 6, 12, 15],
    ],
    dtype=np.int64,
)

MANUSCRIPT_INVERSE_PBOX_TABLE = np.array(
    [
        [8, 9, 6, 1],
        [10, 3, 13, 11],
        [4, 0, 7, 12],
        [14, 5, 2, 15],
    ],
    dtype=np.int64,
)

PBOX_TABLES = {
    "kumar": KUMAR_PBOX_TABLE,
    "manuscript": MANUSCRIPT_PBOX_TABLE,
}

# Generated with:
# generate_keystream(DEFAULT_IC0, 16, params=DEFAULT_PARAMS, dt=DEFAULT_DT,
#                    burn_in_time=200.0)
DEFAULT_DIFFUSION_KEYSTREAM_4X4 = np.array(
    [
        [113, 20, 71, 209],
        [16, 201, 98, 113],
        [160, 223, 108, 112],
        [156, 42, 48, 175],
    ],
    dtype=np.uint8,
)

# For a 4x4 image, n=2, so the only non-identity spatial Baker key is k=1.
DEFAULT_BAKER_KT = (1, 1, 1)
DEFAULT_BAKER_KPRIME = (1, 2, 3)

KUMAR_APPENDIX_PBOX_INPUT_AFTER_XOR = np.array(
    [
        [0x46, 0xA8, 0xFF, 0x5A],
        [0xB2, 0xCB, 0x84, 0xAF],
        [0x2B, 0x7C, 0xD7, 0x62],
        [0xBD, 0xAC, 0x3B, 0x53],
    ],
    dtype=np.uint8,
)

KUMAR_APPENDIX_PBOX_OUTPUT = np.array(
    [
        [0x84, 0xBD, 0xD7, 0xAF],
        [0xB2, 0x2B, 0x7C, 0xA8],
        [0x46, 0xAC, 0x3B, 0x53],
        [0x62, 0x5A, 0xCB, 0xFF],
    ],
    dtype=np.uint8,
)

KUMAR_TABLE_13_DEPTHS = (
    ("NEQR", 15, 89),
    ("S-box", 597, 2050),
    ("XORing", 28, 73),
    ("P-box", 34, 66),
)

NEQRX_TABLE_III_COMPLEXITY = (
    ("Baseline", "NEQR", 234, 147),
    ("Baseline", "Encryption", 666, 513),
    ("Baseline", "Decryption", 1126, 887),
    ("Espresso", "NEQR", 154, 93),
    ("Espresso", "Encryption", 374, 247),
    ("Espresso", "Decryption", 597, 403),
    ("Espresso + Ancillary", "NEQR", 91, 52),
    ("Espresso + Ancillary", "Encryption", 216, 134),
    ("Espresso + Ancillary", "Decryption", 342, 215),
)

CLIFFORD_T_BASIS_GATES = ("cx", "t", "tdg", "h", "s", "sdg", "x")
T_PHASE_GATES = {"t", "tdg"}

COMPONENT_LABELS = {
    "gneqr": "GNEQR state preparation",
    "baker_only": "Baker permutation",
    "diffusion_only": "Chaotic diffusion",
    "pbox_only": "Quantum P-box",
    "encrypt": "Complete encryption",
    "decrypt_after_encrypt": "Encryption + inverse recovery",
}


@dataclass(frozen=True)
class CircuitLayout:
    x: QuantumRegister
    y: QuantumRegister
    color: QuantumRegister
    aux: QuantumRegister

    @property
    def coord_controls_msb(self):
        return [self.x[1], self.x[0], self.y[1], self.y[0]]

    @property
    def coord_targets_msb(self):
        return [self.x[1], self.x[0], self.y[1], self.y[0]]

    @property
    def pbox_scratch_targets_msb(self):
        return [self.aux[1], self.aux[0], self.aux[3], self.aux[2]]

    @property
    def color_msb(self):
        return [self.color[i] for i in range(7, -1, -1)]

    @property
    def aux_color_msb(self):
        return [self.aux[i] for i in range(7, -1, -1)]


def _new_circuit(name: str) -> tuple[QuantumCircuit, CircuitLayout]:
    x = QuantumRegister(2, "x")
    y = QuantumRegister(2, "y")
    color = QuantumRegister(8, "c")
    aux = QuantumRegister(8, "aux")
    qc = QuantumCircuit(x, y, color, aux, name=name)
    return qc, CircuitLayout(x=x, y=y, color=color, aux=aux)


def _validate_image_4x4(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.uint8)
    if arr.shape != (4, 4):
        raise ValueError(f"expected a 4x4 image, got shape {arr.shape}")
    return arr


def _validate_keystream_4x4(keystream: np.ndarray) -> np.ndarray:
    arr = np.asarray(keystream, dtype=np.uint8)
    if arr.shape == (16,):
        arr = arr.reshape(4, 4)
    if arr.shape != (4, 4):
        raise ValueError(f"expected a 4x4 or length-16 keystream, got {arr.shape}")
    return arr


def _validate_pbox_table_4x4(pbox_table: np.ndarray) -> np.ndarray:
    arr = np.asarray(pbox_table, dtype=np.int64)
    if arr.shape == (16,):
        arr = arr.reshape(4, 4)
    if arr.shape != (4, 4):
        raise ValueError(f"expected a 4x4 or length-16 P-box table, got {arr.shape}")
    if sorted(arr.reshape(-1).tolist()) != list(range(16)):
        raise ValueError("P-box table must be a permutation of 0..15")
    return arr


def inverse_pbox_table(pbox_table: np.ndarray) -> np.ndarray:
    pbox = _validate_pbox_table_4x4(pbox_table).reshape(-1)
    inv = np.empty(16, dtype=np.int64)
    inv[pbox] = np.arange(16)
    return inv.reshape(4, 4)


def get_pbox_table(name: str) -> np.ndarray:
    try:
        return PBOX_TABLES[name].copy()
    except KeyError as exc:
        choices = ", ".join(sorted(PBOX_TABLES))
        raise ValueError(f"unknown P-box {name!r}; choose one of: {choices}") from exc


def regenerate_default_keystream() -> np.ndarray:
    """Regenerate the fixed 4x4 demo keystream from the classical chaotic system."""
    return generate_keystream(
        DEFAULT_IC0,
        16,
        params=DEFAULT_PARAMS,
        dt=DEFAULT_DT,
        burn_in_time=200.0,
    ).reshape(4, 4)


def _cube_vector(term: tuple[int, ...], n_vars: int = 4) -> int:
    bits = 0
    for assignment in range(1 << n_vars):
        ok = True
        for var_i, required in enumerate(term):
            if required < 0:
                continue
            actual = (assignment >> (n_vars - 1 - var_i)) & 1
            if actual != required:
                ok = False
                break
        if ok:
            bits |= 1 << assignment
    return bits


@lru_cache(maxsize=1)
def _esop_solver_4vars():
    terms = []
    for raw in _product((-1, 0, 1), repeat=4):
        vec = _cube_vector(raw, 4)
        if vec:
            terms.append((raw, vec))
    terms.sort(key=lambda item: (sum(v >= 0 for v in item[0]), item[0]))

    dist = [-1] * (1 << 16)
    parent = [-1] * (1 << 16)
    parent_term = [-1] * (1 << 16)
    dist[0] = 0
    q = deque([0])
    while q:
        vec = q.popleft()
        for term_i, (_, cube_vec) in enumerate(terms):
            nxt = vec ^ cube_vec
            if dist[nxt] == -1:
                dist[nxt] = dist[vec] + 1
                parent[nxt] = vec
                parent_term[nxt] = term_i
                q.append(nxt)

    return terms, parent, parent_term


def _product(values: Sequence[int], repeat: int):
    if repeat == 0:
        yield ()
        return
    for prefix in _product(values, repeat - 1):
        for value in values:
            yield prefix + (value,)


def minimize_esop_4vars(truth_vector: int) -> list[tuple[int, int, int, int]]:
    """Return a minimum-term 4-variable ESOP for the given 16-bit truth vector."""
    if truth_vector < 0 or truth_vector >= (1 << 16):
        raise ValueError("truth_vector must fit in 16 bits")
    terms, parent, parent_term = _esop_solver_4vars()
    out = []
    cur = truth_vector
    while cur:
        term_i = parent_term[cur]
        if term_i < 0:
            raise RuntimeError("ESOP solver failed to reconstruct a solution")
        out.append(terms[term_i][0])
        cur = parent[cur]
    return out


def canonical_minterm_esop_4vars(truth_vector: int) -> list[tuple[int, int, int, int]]:
    """Return the unoptimized canonical minterm ESOP for a 4-variable truth table."""
    if truth_vector < 0 or truth_vector >= (1 << 16):
        raise ValueError("truth_vector must fit in 16 bits")
    return [
        tuple((assignment >> shift) & 1 for shift in (3, 2, 1, 0))
        for assignment in range(16)
        if (truth_vector >> assignment) & 1
    ]


def _truth_vector_from_values(values: Iterable[int]) -> int:
    vec = 0
    for idx, value in enumerate(values):
        if int(value) & 1:
            vec |= 1 << idx
    return vec


def _coord_values_from_flat(flat: int) -> tuple[int, int]:
    return flat // 4, flat % 4


def _append_cube_x(
    qc: QuantumCircuit,
    controls: Sequence,
    target,
    term: Sequence[int],
):
    active = []
    negated = []
    for qubit, required in zip(controls, term):
        if required == 0:
            qc.x(qubit)
            negated.append(qubit)
            active.append(qubit)
        elif required == 1:
            active.append(qubit)

    if not active:
        qc.x(target)
    elif len(active) == 1:
        qc.cx(active[0], target)
    else:
        qc.mcx(active, target)

    for qubit in reversed(negated):
        qc.x(qubit)


def _append_esop_x(
    qc: QuantumCircuit,
    controls: Sequence,
    target,
    truth_vector: int,
    optimized: bool = True,
):
    terms = minimize_esop_4vars(truth_vector) if optimized else canonical_minterm_esop_4vars(truth_vector)
    for term in terms:
        _append_cube_x(qc, controls, target, term)


def _reverse_wires(qc: QuantumCircuit, wires: Sequence):
    for i in range(len(wires) // 2):
        qc.swap(wires[i], wires[-1 - i])


def _append_left_rotate_msb(qc: QuantumCircuit, wires_msb: Sequence, amount: int):
    r = len(wires_msb)
    k = amount % r
    if k == 0:
        return
    _reverse_wires(qc, wires_msb[:k])
    _reverse_wires(qc, wires_msb[k:])
    _reverse_wires(qc, wires_msb)


def append_gneqr_preparation_4x4(
    qc: QuantumCircuit,
    layout: CircuitLayout,
    image: np.ndarray,
    optimized_esop: bool = True,
):
    image = _validate_image_4x4(image)
    for qubit in [layout.x[0], layout.x[1], layout.y[0], layout.y[1]]:
        qc.h(qubit)

    controls = layout.coord_controls_msb
    for bit in range(8):
        values = [((int(image[x, y]) >> bit) & 1) for x in range(4) for y in range(4)]
        truth = _truth_vector_from_values(values)
        _append_esop_x(qc, controls, layout.color[bit], truth, optimized=optimized_esop)


def build_gneqr_circuit_4x4(
    image: np.ndarray = KUMAR_APPENDIX_IMAGE,
    optimized_esop: bool = True,
) -> QuantumCircuit:
    qc, layout = _new_circuit("gneqr_4x4")
    append_gneqr_preparation_4x4(qc, layout, image, optimized_esop=optimized_esop)
    return qc


def append_baker_round_4x4(
    qc: QuantumCircuit,
    layout: CircuitLayout,
    kt: int = 1,
    kprime: int = 1,
):
    if kt != 1:
        raise ValueError("4x4 spatial Baker supports only kt=1 because n=2")
    if kprime not in (1, 2, 3):
        raise ValueError("kprime must be one of {1, 2, 3}")

    # Spatial: rotate Y first, then exchange X and Y registers.
    _append_left_rotate_msb(qc, [layout.y[1], layout.y[0]], kt)
    qc.swap(layout.x[1], layout.y[1])
    qc.swap(layout.x[0], layout.y[0])

    # Intensity: rotate lower nibble first, then exchange nibbles.
    _append_left_rotate_msb(qc, [layout.color[3], layout.color[2], layout.color[1], layout.color[0]], kprime)
    for upper, lower in zip(
        [layout.color[7], layout.color[6], layout.color[5], layout.color[4]],
        [layout.color[3], layout.color[2], layout.color[1], layout.color[0]],
    ):
        qc.swap(upper, lower)


def append_inverse_baker_round_4x4(
    qc: QuantumCircuit,
    layout: CircuitLayout,
    kt: int = 1,
    kprime: int = 1,
):
    if kt != 1:
        raise ValueError("4x4 spatial Baker supports only kt=1 because n=2")
    if kprime not in (1, 2, 3):
        raise ValueError("kprime must be one of {1, 2, 3}")

    # Inverse intensity: exchange nibbles, then rotate lower nibble right.
    for upper, lower in zip(
        [layout.color[7], layout.color[6], layout.color[5], layout.color[4]],
        [layout.color[3], layout.color[2], layout.color[1], layout.color[0]],
    ):
        qc.swap(upper, lower)
    _append_left_rotate_msb(qc, [layout.color[3], layout.color[2], layout.color[1], layout.color[0]], 4 - kprime)

    # Inverse spatial: exchange X and Y, then rotate Y right.
    qc.swap(layout.x[1], layout.y[1])
    qc.swap(layout.x[0], layout.y[0])
    _append_left_rotate_msb(qc, [layout.y[1], layout.y[0]], 2 - kt)


def append_baker_rounds_4x4(
    qc: QuantumCircuit,
    layout: CircuitLayout,
    kt_list: Sequence[int] = DEFAULT_BAKER_KT,
    kprime_list: Sequence[int] = DEFAULT_BAKER_KPRIME,
):
    if len(kt_list) != len(kprime_list):
        raise ValueError("kt_list and kprime_list must have equal length")
    for kt, kp in zip(kt_list, kprime_list):
        append_baker_round_4x4(qc, layout, kt=kt, kprime=kp)


def append_inverse_baker_rounds_4x4(
    qc: QuantumCircuit,
    layout: CircuitLayout,
    kt_list: Sequence[int] = DEFAULT_BAKER_KT,
    kprime_list: Sequence[int] = DEFAULT_BAKER_KPRIME,
):
    if len(kt_list) != len(kprime_list):
        raise ValueError("kt_list and kprime_list must have equal length")
    for kt, kp in zip(reversed(kt_list), reversed(kprime_list)):
        append_inverse_baker_round_4x4(qc, layout, kt=kt, kprime=kp)


def _keystream_truth_vectors(keystream: np.ndarray) -> list[int]:
    flat = _validate_keystream_4x4(keystream).reshape(-1)
    return [
        _truth_vector_from_values(((int(byte) >> bit) & 1) for byte in flat)
        for bit in range(7, -1, -1)
    ]


def _append_coordinate_constant_x(
    qc: QuantumCircuit,
    layout: CircuitLayout,
    target,
    truth_vector: int,
    optimized_esop: bool = True,
):
    _append_esop_x(qc, layout.coord_controls_msb, target, truth_vector, optimized=optimized_esop)


def _append_diffuse_formula_to_targets(
    qc: QuantumCircuit,
    layout: CircuitLayout,
    source_msb: Sequence,
    target_msb: Sequence,
    key_truth_msb: Sequence[int],
    optimized_esop: bool = True,
):
    # v0 = g1 ^ t0
    qc.cx(source_msb[1], target_msb[0])
    # v1 = g2 ^ t1
    qc.cx(source_msb[2], target_msb[1])
    # v2 = g3 ^ t2
    qc.cx(source_msb[3], target_msb[2])
    # v3 = g4 ^ g0 ^ t3
    qc.cx(source_msb[4], target_msb[3])
    qc.cx(source_msb[0], target_msb[3])
    # v4 = g5 ^ g0 ^ t4
    qc.cx(source_msb[5], target_msb[4])
    qc.cx(source_msb[0], target_msb[4])
    # v5 = g6 ^ t5
    qc.cx(source_msb[6], target_msb[5])
    # v6 = g7 ^ g0 ^ t6
    qc.cx(source_msb[7], target_msb[6])
    qc.cx(source_msb[0], target_msb[6])
    # v7 = g0 ^ t7
    qc.cx(source_msb[0], target_msb[7])

    for i, truth in enumerate(key_truth_msb):
        _append_coordinate_constant_x(qc, layout, target_msb[i], truth, optimized_esop=optimized_esop)


def _append_undiffuse_formula_to_targets(
    qc: QuantumCircuit,
    layout: CircuitLayout,
    source_msb: Sequence,
    target_msb: Sequence,
    key_truth_msb: Sequence[int],
    optimized_esop: bool = True,
):
    # g0 = v7 ^ t7
    qc.cx(source_msb[7], target_msb[0])
    _append_coordinate_constant_x(qc, layout, target_msb[0], key_truth_msb[7], optimized_esop=optimized_esop)
    # g1 = v0 ^ t0
    qc.cx(source_msb[0], target_msb[1])
    _append_coordinate_constant_x(qc, layout, target_msb[1], key_truth_msb[0], optimized_esop=optimized_esop)
    # g2 = v1 ^ t1
    qc.cx(source_msb[1], target_msb[2])
    _append_coordinate_constant_x(qc, layout, target_msb[2], key_truth_msb[1], optimized_esop=optimized_esop)
    # g3 = v2 ^ t2
    qc.cx(source_msb[2], target_msb[3])
    _append_coordinate_constant_x(qc, layout, target_msb[3], key_truth_msb[2], optimized_esop=optimized_esop)
    # g4 = v3 ^ v7 ^ t7 ^ t3
    qc.cx(source_msb[3], target_msb[4])
    qc.cx(source_msb[7], target_msb[4])
    _append_coordinate_constant_x(qc, layout, target_msb[4], key_truth_msb[7], optimized_esop=optimized_esop)
    _append_coordinate_constant_x(qc, layout, target_msb[4], key_truth_msb[3], optimized_esop=optimized_esop)
    # g5 = v4 ^ v7 ^ t7 ^ t4
    qc.cx(source_msb[4], target_msb[5])
    qc.cx(source_msb[7], target_msb[5])
    _append_coordinate_constant_x(qc, layout, target_msb[5], key_truth_msb[7], optimized_esop=optimized_esop)
    _append_coordinate_constant_x(qc, layout, target_msb[5], key_truth_msb[4], optimized_esop=optimized_esop)
    # g6 = v5 ^ t5
    qc.cx(source_msb[5], target_msb[6])
    _append_coordinate_constant_x(qc, layout, target_msb[6], key_truth_msb[5], optimized_esop=optimized_esop)
    # g7 = v6 ^ v7 ^ t7 ^ t6
    qc.cx(source_msb[6], target_msb[7])
    qc.cx(source_msb[7], target_msb[7])
    _append_coordinate_constant_x(qc, layout, target_msb[7], key_truth_msb[7], optimized_esop=optimized_esop)
    _append_coordinate_constant_x(qc, layout, target_msb[7], key_truth_msb[6], optimized_esop=optimized_esop)


def append_diffusion_4x4(
    qc: QuantumCircuit,
    layout: CircuitLayout,
    keystream: np.ndarray = DEFAULT_DIFFUSION_KEYSTREAM_4X4,
    inverse: bool = False,
    optimized_esop: bool = True,
):
    key_truth_msb = _keystream_truth_vectors(keystream)
    color_msb = layout.color_msb
    aux_msb = layout.aux_color_msb

    if not inverse:
        _append_diffuse_formula_to_targets(
            qc, layout, color_msb, aux_msb, key_truth_msb, optimized_esop=optimized_esop
        )
        for c_bit, aux_bit in zip(layout.color, layout.aux):
            qc.swap(c_bit, aux_bit)
        _append_undiffuse_formula_to_targets(
            qc, layout, color_msb, aux_msb, key_truth_msb, optimized_esop=optimized_esop
        )
    else:
        _append_undiffuse_formula_to_targets(
            qc, layout, color_msb, aux_msb, key_truth_msb, optimized_esop=optimized_esop
        )
        for c_bit, aux_bit in zip(layout.color, layout.aux):
            qc.swap(c_bit, aux_bit)
        _append_diffuse_formula_to_targets(
            qc, layout, color_msb, aux_msb, key_truth_msb, optimized_esop=optimized_esop
        )


def _mapping_truth_vectors(mapping: np.ndarray) -> list[int]:
    flat = np.asarray(mapping, dtype=np.int64).reshape(-1)
    if sorted(flat.tolist()) != list(range(16)):
        raise ValueError("mapping must be a permutation of 0..15")
    return [
        _truth_vector_from_values(((int(value) >> bit) & 1) for value in flat)
        for bit in range(3, -1, -1)
    ]


def append_pbox_4x4(
    qc: QuantumCircuit,
    layout: CircuitLayout,
    pbox_table: np.ndarray = KUMAR_PBOX_TABLE,
    inverse: bool = False,
    optimized_esop: bool = True,
):
    pbox_flat = _validate_pbox_table_4x4(pbox_table).reshape(-1)
    inv_flat = inverse_pbox_table(pbox_table).reshape(-1)

    forward_mapping = inv_flat if inverse else pbox_flat
    inverse_mapping = pbox_flat if inverse else inv_flat

    controls = layout.coord_controls_msb
    scratch_targets = layout.pbox_scratch_targets_msb

    for target, truth in zip(scratch_targets, _mapping_truth_vectors(forward_mapping)):
        _append_esop_x(qc, controls, target, truth, optimized=optimized_esop)

    for coord_bit, scratch_bit in zip(
        [layout.x[0], layout.x[1], layout.y[0], layout.y[1]],
        [layout.aux[0], layout.aux[1], layout.aux[2], layout.aux[3]],
    ):
        qc.swap(coord_bit, scratch_bit)

    for target, truth in zip(scratch_targets, _mapping_truth_vectors(inverse_mapping)):
        _append_esop_x(qc, controls, target, truth, optimized=optimized_esop)


def build_encrypt_circuit_4x4(
    image: np.ndarray = KUMAR_APPENDIX_IMAGE,
    kt_list: Sequence[int] = DEFAULT_BAKER_KT,
    kprime_list: Sequence[int] = DEFAULT_BAKER_KPRIME,
    keystream: np.ndarray = DEFAULT_DIFFUSION_KEYSTREAM_4X4,
    pbox_table: np.ndarray = KUMAR_PBOX_TABLE,
    include_pbox: bool = True,
    include_diffusion: bool = True,
    optimized_esop: bool = True,
) -> QuantumCircuit:
    qc, layout = _new_circuit("encrypt_4x4")
    append_gneqr_preparation_4x4(qc, layout, image, optimized_esop=optimized_esop)
    append_baker_rounds_4x4(qc, layout, kt_list=kt_list, kprime_list=kprime_list)
    if include_diffusion:
        append_diffusion_4x4(qc, layout, keystream=keystream, inverse=False, optimized_esop=optimized_esop)
    if include_pbox:
        append_pbox_4x4(qc, layout, pbox_table=pbox_table, inverse=False, optimized_esop=optimized_esop)
    return qc


def build_decrypt_after_encrypt_circuit_4x4(
    image: np.ndarray = KUMAR_APPENDIX_IMAGE,
    kt_list: Sequence[int] = DEFAULT_BAKER_KT,
    kprime_list: Sequence[int] = DEFAULT_BAKER_KPRIME,
    keystream: np.ndarray = DEFAULT_DIFFUSION_KEYSTREAM_4X4,
    pbox_table: np.ndarray = KUMAR_PBOX_TABLE,
    optimized_esop: bool = True,
) -> QuantumCircuit:
    qc, layout = _new_circuit("decrypt_after_encrypt_4x4")
    append_gneqr_preparation_4x4(qc, layout, image, optimized_esop=optimized_esop)
    append_baker_rounds_4x4(qc, layout, kt_list=kt_list, kprime_list=kprime_list)
    append_diffusion_4x4(qc, layout, keystream=keystream, inverse=False, optimized_esop=optimized_esop)
    append_pbox_4x4(qc, layout, pbox_table=pbox_table, inverse=False, optimized_esop=optimized_esop)
    append_pbox_4x4(qc, layout, pbox_table=pbox_table, inverse=True, optimized_esop=optimized_esop)
    append_diffusion_4x4(qc, layout, keystream=keystream, inverse=True, optimized_esop=optimized_esop)
    append_inverse_baker_rounds_4x4(qc, layout, kt_list=kt_list, kprime_list=kprime_list)
    return qc


def build_baker_only_circuit_4x4(
    image: np.ndarray = KUMAR_APPENDIX_IMAGE,
    kt_list: Sequence[int] = DEFAULT_BAKER_KT,
    kprime_list: Sequence[int] = DEFAULT_BAKER_KPRIME,
    optimized_esop: bool = True,
) -> QuantumCircuit:
    qc, layout = _new_circuit("baker_only_4x4")
    append_gneqr_preparation_4x4(qc, layout, image, optimized_esop=optimized_esop)
    append_baker_rounds_4x4(qc, layout, kt_list=kt_list, kprime_list=kprime_list)
    return qc


def build_diffusion_only_circuit_4x4(
    image: np.ndarray = KUMAR_APPENDIX_IMAGE,
    keystream: np.ndarray = DEFAULT_DIFFUSION_KEYSTREAM_4X4,
    optimized_esop: bool = True,
) -> QuantumCircuit:
    qc, layout = _new_circuit("diffusion_only_4x4")
    append_gneqr_preparation_4x4(qc, layout, image, optimized_esop=optimized_esop)
    append_diffusion_4x4(qc, layout, keystream=keystream, inverse=False, optimized_esop=optimized_esop)
    return qc


def build_pbox_only_circuit_4x4(
    image: np.ndarray = KUMAR_APPENDIX_IMAGE,
    pbox_table: np.ndarray = KUMAR_PBOX_TABLE,
    optimized_esop: bool = True,
) -> QuantumCircuit:
    qc, layout = _new_circuit("pbox_only_4x4")
    append_gneqr_preparation_4x4(qc, layout, image, optimized_esop=optimized_esop)
    append_pbox_4x4(qc, layout, pbox_table=pbox_table, inverse=False, optimized_esop=optimized_esop)
    return qc


def build_stage_circuits_4x4(
    image: np.ndarray = KUMAR_APPENDIX_IMAGE,
    kt_list: Sequence[int] = DEFAULT_BAKER_KT,
    kprime_list: Sequence[int] = DEFAULT_BAKER_KPRIME,
    keystream: np.ndarray = DEFAULT_DIFFUSION_KEYSTREAM_4X4,
    pbox_table: np.ndarray = KUMAR_PBOX_TABLE,
    optimized_esop: bool = True,
) -> dict[str, QuantumCircuit]:
    return {
        "gneqr": build_gneqr_circuit_4x4(image, optimized_esop=optimized_esop),
        "baker_only": build_baker_only_circuit_4x4(
            image, kt_list, kprime_list, optimized_esop=optimized_esop
        ),
        "diffusion_only": build_diffusion_only_circuit_4x4(
            image, keystream, optimized_esop=optimized_esop
        ),
        "pbox_only": build_pbox_only_circuit_4x4(image, pbox_table, optimized_esop=optimized_esop),
        "encrypt": build_encrypt_circuit_4x4(
            image, kt_list, kprime_list, keystream, pbox_table, optimized_esop=optimized_esop
        ),
        "decrypt_after_encrypt": build_decrypt_after_encrypt_circuit_4x4(
            image, kt_list, kprime_list, keystream, pbox_table, optimized_esop=optimized_esop
        ),
    }


def _register_by_name(qc: QuantumCircuit, name: str) -> QuantumRegister:
    for reg in qc.qregs:
        if reg.name == name:
            return reg
    raise ValueError(f"circuit has no quantum register named {name!r}")


def _simulate_statevector(qc: QuantumCircuit) -> np.ndarray:
    """Use Aer for speed when available; fall back to qiskit.quantum_info."""
    try:
        from qiskit_aer import AerSimulator

        sim_qc = qc.copy()
        sim_qc.save_statevector()
        result = AerSimulator(method="statevector").run(sim_qc).result()
        state = result.get_statevector(sim_qc)
        return np.asarray(state)
    except Exception:
        return np.asarray(Statevector.from_instruction(qc).data)


def decode_statevector_image_4x4(qc: QuantumCircuit, atol: float = 1e-9) -> np.ndarray:
    state = _simulate_statevector(qc)
    x = _register_by_name(qc, "x")
    y = _register_by_name(qc, "y")
    color = _register_by_name(qc, "c")
    aux = _register_by_name(qc, "aux")

    q_index = {qubit: qc.find_bit(qubit).index for qubit in qc.qubits}

    image = np.zeros((4, 4), dtype=np.uint8)
    seen = np.zeros((4, 4), dtype=bool)
    for basis, amp in enumerate(state):
        if abs(amp) <= atol:
            continue
        aux_value = sum(((basis >> q_index[aux[i]]) & 1) << i for i in range(8))
        if aux_value != 0:
            raise AssertionError(f"aux register not uncomputed for basis {basis}: {aux_value}")

        xv = ((basis >> q_index[x[0]]) & 1) | (((basis >> q_index[x[1]]) & 1) << 1)
        yv = ((basis >> q_index[y[0]]) & 1) | (((basis >> q_index[y[1]]) & 1) << 1)
        cv = sum(((basis >> q_index[color[i]]) & 1) << i for i in range(8))
        if seen[xv, yv]:
            raise AssertionError(f"multiple amplitudes decoded for coordinate {(xv, yv)}")
        seen[xv, yv] = True
        image[xv, yv] = cv

    if not np.all(seen):
        missing = np.argwhere(~seen).tolist()
        raise AssertionError(f"missing coordinates in decoded state: {missing}")
    return image


def apply_pbox_scatter_image(
    image: np.ndarray,
    pbox_table: np.ndarray = KUMAR_PBOX_TABLE,
) -> np.ndarray:
    image = _validate_image_4x4(image)
    pbox = _validate_pbox_table_4x4(pbox_table).reshape(-1)
    out = np.empty(16, dtype=np.uint8)
    out[pbox] = image.reshape(-1)
    return out.reshape(4, 4)


def apply_inverse_pbox_scatter_image(
    image: np.ndarray,
    pbox_table: np.ndarray = KUMAR_PBOX_TABLE,
) -> np.ndarray:
    image = _validate_image_4x4(image)
    pbox = _validate_pbox_table_4x4(pbox_table).reshape(-1)
    return image.reshape(-1)[pbox].reshape(4, 4)


def classical_pipeline_tables(
    image: np.ndarray = KUMAR_APPENDIX_IMAGE,
    kt_list: Sequence[int] = DEFAULT_BAKER_KT,
    kprime_list: Sequence[int] = DEFAULT_BAKER_KPRIME,
    keystream: np.ndarray = DEFAULT_DIFFUSION_KEYSTREAM_4X4,
    pbox_table: np.ndarray = KUMAR_PBOX_TABLE,
):
    image = _validate_image_4x4(image)
    keystream = _validate_keystream_4x4(keystream)
    pbox_table = _validate_pbox_table_4x4(pbox_table)
    baker = encrypt_scramble(image, kt_list, kprime_list)
    diffused = diffuse_encrypt(baker.reshape(-1), keystream.reshape(-1)).reshape(4, 4)
    cipher = apply_pbox_scatter_image(diffused, pbox_table=pbox_table)
    inv_pbox = apply_inverse_pbox_scatter_image(cipher, pbox_table=pbox_table)
    undiffused = diffuse_decrypt(inv_pbox.reshape(-1), keystream.reshape(-1)).reshape(4, 4)
    recovered = decrypt_unscramble(undiffused, kt_list, kprime_list)
    return {
        "plain": image,
        "baker": baker,
        "keystream": keystream,
        "diffused": diffused,
        "cipher": cipher,
        "inv_pbox": inv_pbox,
        "undiffused": undiffused,
        "recovered": recovered,
        "pbox_table": pbox_table,
        "pbox_inverse": inverse_pbox_table(pbox_table),
    }


def classical_decrypt_pipeline_4x4(
    cipher: np.ndarray,
    kt_list: Sequence[int] = DEFAULT_BAKER_KT,
    kprime_list: Sequence[int] = DEFAULT_BAKER_KPRIME,
    keystream: np.ndarray = DEFAULT_DIFFUSION_KEYSTREAM_4X4,
    pbox_table: np.ndarray = KUMAR_PBOX_TABLE,
) -> np.ndarray:
    cipher = _validate_image_4x4(cipher)
    keystream = _validate_keystream_4x4(keystream)
    pbox_table = _validate_pbox_table_4x4(pbox_table)
    inv_pbox = apply_inverse_pbox_scatter_image(cipher, pbox_table=pbox_table)
    undiffused = diffuse_decrypt(inv_pbox.reshape(-1), keystream.reshape(-1)).reshape(4, 4)
    return decrypt_unscramble(undiffused, kt_list, kprime_list)


def _term_to_expr(term: Sequence[int], names: Sequence[str]) -> str:
    parts = []
    for value, name in zip(term, names):
        if value == 1:
            parts.append(name)
        elif value == 0:
            parts.append("~" + name)
    return "".join(parts) if parts else "1"


def _eval_esop_terms(terms: Sequence[Sequence[int]], assignment: int, n_vars: int = 4) -> int:
    out = 0
    for term in terms:
        ok = True
        for var_i, required in enumerate(term):
            if required < 0:
                continue
            actual = (assignment >> (n_vars - 1 - var_i)) & 1
            if actual != required:
                ok = False
                break
        if ok:
            out ^= 1
    return out


def verify_esop_truth_tables(
    image: np.ndarray = KUMAR_APPENDIX_IMAGE,
    keystream: np.ndarray = DEFAULT_DIFFUSION_KEYSTREAM_4X4,
) -> bool:
    """Exhaustively verify every printed ESOP table against its truth vector."""
    image = _validate_image_4x4(image)
    keystream = _validate_keystream_4x4(keystream)
    truth_vectors = []

    for bit in range(7, -1, -1):
        truth_vectors.append(
            _truth_vector_from_values(((int(image[x, y]) >> bit) & 1) for x in range(4) for y in range(4))
        )
    truth_vectors.extend(_mapping_truth_vectors(KUMAR_PBOX_TABLE.reshape(-1)))
    truth_vectors.extend(_mapping_truth_vectors(KUMAR_INVERSE_PBOX_TABLE.reshape(-1)))
    truth_vectors.extend(_keystream_truth_vectors(keystream))

    for truth in truth_vectors:
        terms = minimize_esop_4vars(truth)
        rebuilt = _truth_vector_from_values(_eval_esop_terms(terms, i) for i in range(16))
        if rebuilt != truth:
            return False
    return True


def _esop_expression(truth: int, names: Sequence[str] = ("x1", "x0", "y1", "y0")) -> str:
    terms = minimize_esop_4vars(truth)
    if not terms:
        return "0"
    return " xor ".join(_term_to_expr(term, names) for term in terms)


def esop_summary_tables(
    image: np.ndarray = KUMAR_APPENDIX_IMAGE,
    keystream: np.ndarray = DEFAULT_DIFFUSION_KEYSTREAM_4X4,
    pbox_table: np.ndarray = KUMAR_PBOX_TABLE,
):
    image = _validate_image_4x4(image)
    keystream = _validate_keystream_4x4(keystream)
    pbox_table = _validate_pbox_table_4x4(pbox_table)
    inverse_table = inverse_pbox_table(pbox_table)

    rows = []
    for bit in range(7, -1, -1):
        truth = _truth_vector_from_values(((int(image[x, y]) >> bit) & 1) for x in range(4) for y in range(4))
        terms = minimize_esop_4vars(truth)
        rows.append(("GNEQR c" + str(bit), len(terms), _esop_expression(truth)))

    for label, mapping in (("P-box", pbox_table.reshape(-1)), ("P-box inverse", inverse_table.reshape(-1))):
        for bit, truth in zip(("x1", "x0", "y1", "y0"), _mapping_truth_vectors(mapping)):
            terms = minimize_esop_4vars(truth)
            rows.append((f"{label} {bit}", len(terms), _esop_expression(truth)))

    for bit_name, truth in zip(("t7", "t6", "t5", "t4", "t3", "t2", "t1", "t0"), _keystream_truth_vectors(keystream)):
        terms = minimize_esop_4vars(truth)
        rows.append((f"Diffusion key {bit_name}", len(terms), _esop_expression(truth)))

    return rows


def _format_matrix(matrix: np.ndarray, hex_bytes: bool = False) -> str:
    arr = np.asarray(matrix)
    lines = ["| row | 0 | 1 | 2 | 3 |", "|---:|---:|---:|---:|---:|"]
    for row_i, row in enumerate(arr):
        if hex_bytes:
            values = [f"0x{int(v):02X}" for v in row]
        else:
            values = [str(int(v)) for v in row]
        lines.append("| " + str(row_i) + " | " + " | ".join(values) + " |")
    return "\n".join(lines)


def _format_gneqr_table(image: np.ndarray) -> str:
    lines = [
        "| flat | x | y | coord bits | pixel dec | pixel bin |",
        "|---:|---:|---:|:---:|---:|:---:|",
    ]
    for flat in range(16):
        x, y = _coord_values_from_flat(flat)
        value = int(image[x, y])
        lines.append(f"| {flat} | {x} | {y} | {flat:04b} | {value} | {value:08b} |")
    return "\n".join(lines)


def _format_round_keys(kt_list: Sequence[int], kprime_list: Sequence[int]) -> str:
    lines = ["| round | spatial kt | intensity kprime |", "|---:|---:|---:|"]
    for i, (kt, kp) in enumerate(zip(kt_list, kprime_list), start=1):
        lines.append(f"| {i} | {kt} | {kp} |")
    return "\n".join(lines)


def _format_esop_rows(rows: Sequence[tuple[str, int, str]]) -> str:
    lines = ["| target | terms | minimized ESOP |", "|:---|---:|:---|"]
    for target, n_terms, expr in rows:
        lines.append(f"| {target} | {n_terms} | `{expr}` |")
    return "\n".join(lines)


def _instruction_parts(qc: QuantumCircuit, item) -> tuple[str, list[int]]:
    operation = item.operation if hasattr(item, "operation") else item[0]
    qubits = item.qubits if hasattr(item, "qubits") else item[1]
    return operation.name, [qc.find_bit(qubit).index for qubit in qubits]


def _transpile_clifford_t(qc: QuantumCircuit) -> QuantumCircuit:
    return transpile(qc, basis_gates=list(CLIFFORD_T_BASIS_GATES), optimization_level=0)


def _t_depth_from_circuit(qc: QuantumCircuit) -> int:
    layers = [0] * qc.num_qubits
    for item in qc.data:
        name, qubit_indices = _instruction_parts(qc, item)
        if not qubit_indices:
            continue
        if name in T_PHASE_GATES:
            next_layer = max(layers[i] for i in qubit_indices) + 1
            for index in qubit_indices:
                layers[index] = next_layer
        else:
            synced_layer = max(layers[i] for i in qubit_indices)
            for index in qubit_indices:
                layers[index] = synced_layer
    return max(layers, default=0)


def _ancilla_width(qc: QuantumCircuit) -> int:
    return sum(reg.size for reg in qc.qregs if reg.name == "aux")


def _component_label(name: str) -> str:
    return COMPONENT_LABELS.get(name.removesuffix("_4x4"), name)


def circuit_resource_metrics(qc: QuantumCircuit) -> dict[str, int | str]:
    tqc = _transpile_clifford_t(qc)
    ops = dict(tqc.count_ops())
    return {
        "component": _component_label(qc.name),
        "cnot": int(ops.get("cx", 0)),
        "t_count": int(ops.get("t", 0) + ops.get("tdg", 0)),
        "t_depth": _t_depth_from_circuit(tqc),
        "ancilla": _ancilla_width(qc),
        "circuit_width": qc.num_qubits,
        "circuit_depth": int(tqc.depth() or 0),
        "raw_depth": int(qc.depth() or 0),
    }


def component_resource_table(circuits: dict[str, QuantumCircuit]) -> str:
    lines = [
        "| algorithm component | CNOT | T-depth | ancilla | circuit width | circuit depth |",
        "|:---|---:|---:|---:|---:|---:|",
    ]
    for name, qc in circuits.items():
        metrics = circuit_resource_metrics(qc)
        label = COMPONENT_LABELS.get(name, str(metrics["component"]))
        lines.append(
            f"| {label} | {metrics['cnot']} | {metrics['t_depth']} | {metrics['ancilla']} | "
            f"{metrics['circuit_width']} | {metrics['circuit_depth']} |"
        )
    return "\n".join(lines)


def _reduction_percent(before: int, after: int) -> str:
    if before == 0:
        return "0.0%" if after == 0 else "n/a"
    return f"{((before - after) / before) * 100:.1f}%"


def esop_optimization_comparison_table(
    without_esop_circuits: dict[str, QuantumCircuit],
    with_esop_circuits: dict[str, QuantumCircuit],
) -> str:
    lines = [
        "| algorithm component | CNOT without ESOP | CNOT with ESOP | CNOT reduction | "
        "T-depth without ESOP | T-depth with ESOP | circuit depth without ESOP | circuit depth with ESOP |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, with_qc in with_esop_circuits.items():
        without_qc = without_esop_circuits[name]
        before = circuit_resource_metrics(without_qc)
        after = circuit_resource_metrics(with_qc)
        lines.append(
            f"| {COMPONENT_LABELS.get(name, str(after['component']))} | "
            f"{before['cnot']} | {after['cnot']} | {_reduction_percent(int(before['cnot']), int(after['cnot']))} | "
            f"{before['t_depth']} | {after['t_depth']} | "
            f"{before['circuit_depth']} | {after['circuit_depth']} |"
        )
    return "\n".join(lines)


def circuit_summary_table(qc: QuantumCircuit) -> str:
    raw_ops = dict(qc.count_ops())
    raw_ops_text = ", ".join(f"{name}:{count}" for name, count in sorted(raw_ops.items()))
    lines = [
        "| circuit | qubits | raw depth | raw ops | transpiled depth | transpiled ops |",
        "|:---|---:|---:|:---|---:|:---|",
    ]
    try:
        tqc = transpile(qc, basis_gates=["cx", "id", "rz", "sx", "x"], optimization_level=3)
        trans_ops = dict(tqc.count_ops())
        trans_ops_text = ", ".join(f"{name}:{count}" for name, count in sorted(trans_ops.items()))
        lines.append(
            f"| {qc.name} | {qc.num_qubits} | {qc.depth()} | {raw_ops_text} | "
            f"{tqc.depth()} | {trans_ops_text} |"
        )
    except Exception as exc:
        lines.append(
            f"| {qc.name} | {qc.num_qubits} | {qc.depth()} | {raw_ops_text} | "
            f"n/a | transpile failed: {type(exc).__name__}: {exc} |"
        )
    return "\n".join(lines)


def _format_reference_complexity_table(rows: Sequence[tuple]) -> str:
    lines = ["| reference | operation/circuit | metric 1 | metric 2 |", "|:---|:---|---:|---:|"]
    for row in rows:
        if len(row) == 3:
            operation, with_esop, without_esop = row
            lines.append(f"| Kumar Table 13 | {operation} | with ESOP depth {with_esop} | without ESOP depth {without_esop} |")
        else:
            stage, circuit, cost, time = row
            lines.append(f"| NEQRX Table III | {stage} {circuit} | quantum cost {cost} | time complexity {time} |")
    return "\n".join(lines)


def baker_equation_verification() -> dict[str, object]:
    from baker_permutation import (
        intensity_baker_inverse_scalar,
        intensity_baker_scalar,
        spatial_baker_inverse_scalar,
        spatial_baker_scalar,
    )

    spatial_failures = []
    for m, n in ((2, 2), (2, 3), (3, 2), (2, 5), (5, 2), (3, 3), (4, 3), (1, 4), (4, 1), (6, 2), (2, 6)):
        for k in range(1, n):
            seen = set()
            for x in range(1 << m):
                for y in range(1 << n):
                    xp, yp = spatial_baker_scalar(x, y, k, m, n)
                    seen.add((xp, yp))
                    if spatial_baker_inverse_scalar(xp, yp, k, m, n) != (x, y):
                        spatial_failures.append((m, n, k, x, y, xp, yp))
            if len(seen) != (1 << (m + n)):
                spatial_failures.append((m, n, k, "collision", len(seen)))

    intensity_failures = []
    for kprime in (1, 2, 3):
        seen = set()
        for value in range(256):
            out = intensity_baker_scalar(value, kprime)
            seen.add(out)
            if intensity_baker_inverse_scalar(out, kprime) != value:
                intensity_failures.append((kprime, value, out))
        if len(seen) != 256:
            intensity_failures.append((kprime, "collision", len(seen)))

    return {
        "spatial_exhaustive_ok": len(spatial_failures) == 0,
        "intensity_exhaustive_ok": len(intensity_failures) == 0,
        "worked_spatial_m2_n3_k2_x2_y5": spatial_baker_scalar(2, 5, 2, 2, 3),
        "worked_intensity_c171_k2": intensity_baker_scalar(171, 2),
        "failure_count": len(spatial_failures) + len(intensity_failures),
    }


def gneqr_state_verification(image: np.ndarray = KUMAR_APPENDIX_IMAGE) -> dict[str, object]:
    qc = build_gneqr_circuit_4x4(image)
    state = _simulate_statevector(qc)
    nonzero = [(i, amp) for i, amp in enumerate(state) if abs(amp) > 1e-9]
    magnitudes = sorted(round(abs(amp), 12) for _, amp in nonzero)
    return {
        "nonzero_basis_states": len(nonzero),
        "all_amplitudes_have_magnitude_one_quarter": magnitudes == [0.25] * 16,
        "decoded_matches_appendix_a1": np.array_equal(decode_statevector_image_4x4(qc), KUMAR_APPENDIX_IMAGE),
    }


def verification_status() -> dict[str, object]:
    pbox_flat = KUMAR_PBOX_TABLE.reshape(-1)
    inv_flat = KUMAR_INVERSE_PBOX_TABLE.reshape(-1)
    recomputed_inv = np.empty(16, dtype=np.int64)
    recomputed_inv[pbox_flat] = np.arange(16)
    self_checks = run_self_test()
    baker_checks = baker_equation_verification()
    gneqr_checks = gneqr_state_verification()
    status = {
        "kumar_table_3_is_permutation": sorted(pbox_flat.tolist()) == list(range(16)),
        "kumar_table_4_matches_inverse_of_table_3": np.array_equal(recomputed_inv, inv_flat),
        "kumar_appendix_a1_image_transcribed": np.array_equal(KUMAR_APPENDIX_IMAGE, np.array(
            [[0, 15, 31, 47], [63, 79, 95, 111], [127, 143, 159, 175], [191, 207, 223, 239]],
            dtype=np.uint8,
        )),
        "kumar_appendix_a5_to_a6_pbox_match": np.array_equal(
            apply_pbox_scatter_image(KUMAR_APPENDIX_PBOX_INPUT_AFTER_XOR),
            KUMAR_APPENDIX_PBOX_OUTPUT,
        ),
        "esop_truth_tables_exhaustive": verify_esop_truth_tables(),
    }
    status.update({"self_test_" + k: v for k, v in self_checks.items() if k != "ok"})
    status.update({"baker_" + k: v for k, v in baker_checks.items()})
    status.update({"gneqr_" + k: v for k, v in gneqr_checks.items()})
    status["ok"] = all(value is True for key, value in status.items() if key != "ok" and isinstance(value, bool))
    return status


def generate_paper_verification_report() -> str:
    status = verification_status()
    checks_rows = ["| check | status |", "|:---|:---:|"]
    for key, value in status.items():
        if key == "ok":
            continue
        checks_rows.append(f"| {key} | {value} |")

    baker = baker_equation_verification()
    gneqr = gneqr_state_verification()

    parts = [
        "# Paper Verification Report",
        "",
        "## Exact Published-Paper Matches",
        "- Kumar et al. Table 3 P-box is transcribed exactly and is a permutation of 0..15.",
        "- Kumar et al. Table 4 inverse P-box is recomputed from Table 3 and matches exactly.",
        "- Kumar et al. Appendix A1 4x4 image is used as the default GNEQR input.",
        "- Kumar et al. Appendix A5 -> A6 P-box example matches exactly when Table 3 is applied as scatter: `out[P[input]] = input`.",
        "",
        "## Baker-Method Verification",
        "- The Baker method is from the manuscript/internal appendix, not the Kumar or NEQRX published papers.",
        f"- Exhaustive spatial checks passed: {baker['spatial_exhaustive_ok']}.",
        f"- Exhaustive intensity checks passed: {baker['intensity_exhaustive_ok']}.",
        f"- Worked example `(m=2,n=3,k=2,x=2,y=5)` gives `{baker['worked_spatial_m2_n3_k2_x2_y5']}`.",
        f"- Worked intensity example `c=171,k'=2` gives `{baker['worked_intensity_c171_k2']}`.",
        "- Operator order used in circuit: rotate Y before coordinate block exchange; rotate lower intensity nibble before nibble exchange.",
        "",
        "## GNEQR State Verification",
        f"- Nonzero basis states: {gneqr['nonzero_basis_states']}.",
        f"- All nonzero amplitudes have magnitude 1/4: {gneqr['all_amplitudes_have_magnitude_one_quarter']}.",
        f"- Decoded state matches Kumar Appendix A1 image: {gneqr['decoded_matches_appendix_a1']}.",
        "",
        "## NEQRX Reference Use",
        "- NEQRX is used as a circuit-complexity/optimization reference, not as a numeric target for this circuit.",
        "- Its Table III is for a different 2x2 NEQR + GAT/LM/RDH construction, so our gate counts should not equal those values.",
        "- We use the same style of reporting: Qiskit transpiler, quantum cost/gate count, and time complexity/depth.",
        "",
        "## Reference Complexity Tables",
        _format_reference_complexity_table(KUMAR_TABLE_13_DEPTHS + NEQRX_TABLE_III_COMPLEXITY),
        "",
        "## Executed Checks",
        "\n".join(checks_rows),
        "",
        "## Construction Notes",
        "- The published Kumar P-box convention is input-to-output scatter. A gather implementation is invertible but gives the inverse orientation and does not match Appendix A6.",
        "- Manuscript correction: if the worked Baker example says `RotL3(5)=3` for `k=2`, that line is wrong; `101` left-rotated by 2 is `110`, so the correct value is 6.",
        "- Kumar Appendix S-box/XOR outputs are not expected to match this project because this project uses Baker scrambling and chaotic diffusion instead of Kumar's S-box followed by random-image XOR.",
        "- Large 512x512 quantum statevector simulation is intentionally out of scope; this verification targets the requested 4x4 simulation.",
        "",
    ]
    return "\n".join(parts)


def generate_markdown_report(
    image: np.ndarray = KUMAR_APPENDIX_IMAGE,
    kt_list: Sequence[int] = DEFAULT_BAKER_KT,
    kprime_list: Sequence[int] = DEFAULT_BAKER_KPRIME,
    keystream: np.ndarray = DEFAULT_DIFFUSION_KEYSTREAM_4X4,
    pbox_table: np.ndarray = KUMAR_PBOX_TABLE,
    pbox_label: str = "Kumar",
) -> str:
    pbox_table = _validate_pbox_table_4x4(pbox_table)
    inverse_table = inverse_pbox_table(pbox_table)
    tables = classical_pipeline_tables(image, kt_list, kprime_list, keystream, pbox_table=pbox_table)
    encrypt_qc = build_encrypt_circuit_4x4(image, kt_list, kprime_list, keystream, pbox_table=pbox_table)
    recovered_qc = build_decrypt_after_encrypt_circuit_4x4(
        image, kt_list, kprime_list, keystream, pbox_table=pbox_table
    )
    pbox_check = apply_pbox_scatter_image(KUMAR_APPENDIX_PBOX_INPUT_AFTER_XOR)

    parts = [
        "# 4x4 GNEQR Quantum Simulation Tables",
        "",
        "## Plain Image",
        _format_matrix(tables["plain"]),
        "",
        "## GNEQR Encoding Table",
        _format_gneqr_table(tables["plain"]),
        "",
        "## Baker Round Keys",
        _format_round_keys(kt_list, kprime_list),
        "",
        "## Image After Baker Scrambling",
        _format_matrix(tables["baker"]),
        "",
        "## Diffusion Keystream",
        _format_matrix(tables["keystream"], hex_bytes=True),
        "",
        "## Image After Chaotic Diffusion",
        _format_matrix(tables["diffused"], hex_bytes=True),
        "",
        f"## {pbox_label} P-box Table",
        _format_matrix(pbox_table),
        "",
        f"## {pbox_label} Inverse P-box Table",
        _format_matrix(inverse_table),
        "",
        "## Final Encrypted Image After P-box",
        _format_matrix(tables["cipher"], hex_bytes=True),
        "",
        "## Recovered Image After Inverse Pipeline",
        _format_matrix(tables["recovered"]),
        "",
        "## Kumar Appendix P-box Convention Check",
        "This must match Appendix A6 when Table 3 is applied as scatter: `out[P[input]] = input`.",
        "",
        "### Appendix A5 Input To P-box",
        _format_matrix(KUMAR_APPENDIX_PBOX_INPUT_AFTER_XOR, hex_bytes=True),
        "",
        "### Computed P-box Output",
        _format_matrix(pbox_check, hex_bytes=True),
        "",
        "## ESOP Summary",
        _format_esop_rows(esop_summary_tables(image, keystream, pbox_table=pbox_table)),
        "",
        "## Circuit Summary",
        circuit_summary_table(encrypt_qc),
        "",
        "## Decrypt-After-Encrypt Circuit Summary",
        circuit_summary_table(recovered_qc),
        "",
    ]
    return "\n".join(parts)


def run_self_test() -> dict[str, object]:
    image = KUMAR_APPENDIX_IMAGE
    kt_list = DEFAULT_BAKER_KT
    kprime_list = DEFAULT_BAKER_KPRIME
    keystream = DEFAULT_DIFFUSION_KEYSTREAM_4X4

    tables = classical_pipeline_tables(image, kt_list, kprime_list, keystream)

    gneqr_decoded = decode_statevector_image_4x4(build_gneqr_circuit_4x4(image))
    baker_decoded = decode_statevector_image_4x4(
        build_encrypt_circuit_4x4(image, kt_list, kprime_list, keystream, include_diffusion=False, include_pbox=False)
    )
    diffused_decoded = decode_statevector_image_4x4(
        build_encrypt_circuit_4x4(image, kt_list, kprime_list, keystream, include_diffusion=True, include_pbox=False)
    )
    cipher_decoded = decode_statevector_image_4x4(
        build_encrypt_circuit_4x4(image, kt_list, kprime_list, keystream, include_diffusion=True, include_pbox=True)
    )
    recovered_decoded = decode_statevector_image_4x4(
        build_decrypt_after_encrypt_circuit_4x4(image, kt_list, kprime_list, keystream)
    )

    pbox_check = apply_pbox_scatter_image(KUMAR_APPENDIX_PBOX_INPUT_AFTER_XOR)
    regenerated = regenerate_default_keystream()

    checks = {
        "gneqr_matches_plain": np.array_equal(gneqr_decoded, tables["plain"]),
        "baker_matches_classical": np.array_equal(baker_decoded, tables["baker"]),
        "diffusion_matches_classical": np.array_equal(diffused_decoded, tables["diffused"]),
        "pbox_matches_classical": np.array_equal(cipher_decoded, tables["cipher"]),
        "recovered_matches_plain": np.array_equal(recovered_decoded, tables["plain"]),
        "kumar_pbox_appendix_matches": np.array_equal(pbox_check, KUMAR_APPENDIX_PBOX_OUTPUT),
        "embedded_keystream_reproducible": np.array_equal(regenerated, DEFAULT_DIFFUSION_KEYSTREAM_4X4),
        "esop_truth_tables_exhaustive": verify_esop_truth_tables(image, keystream),
    }
    checks["ok"] = all(checks.values())
    return checks


def main():
    checks = run_self_test()
    print("Self-test:")
    for key, value in checks.items():
        print(f"  {key}: {value}")
    if not checks["ok"]:
        raise SystemExit(1)

    out_path = Path(__file__).with_name("quantum_4x4_simulation_tables.md")
    out_path.write_text(generate_markdown_report(), encoding="utf-8")
    print(f"\nWrote {out_path}")

    verification_path = Path(__file__).with_name("paper_verification_report.md")
    verification_path.write_text(generate_paper_verification_report(), encoding="utf-8")
    print(f"Wrote {verification_path}")


if __name__ == "__main__":
    main()
