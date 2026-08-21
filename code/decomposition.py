"""Decomposition toolbox. Objective: an edge is satisfied when its two
endpoints take different values (0/1). Quality = satisfied edge count.
cut -> solve pieces in sequence -> stitch one full answer.
solve_exact gives ground truth at small sizes by brute force."""

import random as _random
from itertools import product
import minorminer


def score_assignment(problem, assignment):
    """Number of edges whose endpoints differ under this assignment."""
    return sum(1 for u, v in problem.edges
               if assignment[u] != assignment[v])


def solve_exact(problem):
    """Try every possible assignment. Returns (best_score, one best answer).
    Brute force: 2^n combinations, so small problems only."""
    nodes = sorted(problem.nodes)
    if len(nodes) > 20:
        raise ValueError("solve_exact is brute force; too many variables")
    best_score, best_assignment = -1, None
    for bits in product([0, 1], repeat=len(nodes)):
        assignment = dict(zip(nodes, bits))
        s = score_assignment(problem, assignment)
        if s > best_score:
            best_score, best_assignment = s, assignment
    return best_score, best_assignment


def cut(problem, piece_size, overlap=1, order_seed=None, ordering="index"):
    """Slice variables into consecutive overlapping pieces.
    ordering: "index" (sorted order, original behaviour),
              "shuffle" (random via order_seed, the old lottery),
              "spectral" (structure-aware, deterministic)."""
    if piece_size <= overlap:
        raise ValueError("piece_size must exceed overlap")
    if ordering == "spectral":
        nodes = spectral_order(problem)
    else:
        nodes = sorted(problem.nodes)
        if ordering == "shuffle" and order_seed is not None:
            _random.Random(order_seed).shuffle(nodes)
    pieces, i = [], 0
    while True:
        piece = nodes[i:i + piece_size]
        pieces.append(piece)
        if piece[-1] == nodes[-1]:
            return pieces
        i += piece_size - overlap


def solve_piece(problem, piece, fixed):
    """Brute-force the piece's undecided variables, respecting earlier
    decisions. Scores every edge whose two endpoints both have values.
    Returns the chosen values for the undecided variables only."""
    free = [v for v in piece if v not in fixed]
    if not free:
        return {}
    best_score, best_values = -1, None
    for bits in product([0, 1], repeat=len(free)):
        trial = dict(fixed)
        trial.update(zip(free, bits))
        s = sum(1 for u, v in problem.edges
                if u in trial and v in trial and trial[u] != trial[v])
        if s > best_score:
            best_score, best_values = s, dict(zip(free, bits))
    return best_values


def stitch_solve(problem, piece_size, overlap=1, order_seed=None, ordering="index"):
    """Cut, solve pieces in order, assemble one full answer."""
    pieces = cut(problem, piece_size, overlap, order_seed,
                 ordering="shuffle" if order_seed is not None and ordering == "index" else ordering)
    assignment = {}
    for piece in pieces:
        assignment.update(solve_piece(problem, piece, assignment))
    return score_assignment(problem, assignment), assignment, pieces


def qbsolv_style_solve(problem, piece_size, seed=0, max_rounds=20):
    """Iterative subset improvement in the style of QBSolv (Booth et al.):
    start from a random assignment, repeatedly re-solve random subsets of
    piece_size variables exactly (rest frozen), keep improvements, stop
    when a full round yields none. Returns (score, assignment, rounds)."""
    rng = _random.Random(seed)
    nodes = sorted(problem.nodes)
    assignment = {v: rng.randint(0, 1) for v in nodes}
    best = score_assignment(problem, assignment)
    for round_no in range(1, max_rounds + 1):
        improved = False
        order = nodes[:]
        rng.shuffle(order)
        for i in range(0, len(order), piece_size):
            subset = order[i:i + piece_size]
            fixed = {v: x for v, x in assignment.items() if v not in subset}
            trial = dict(fixed)
            trial.update(solve_piece(problem, subset, fixed))
            s = score_assignment(problem, trial)
            if s > best:
                assignment, best, improved = trial, s, True
        if not improved:
            return best, assignment, round_no
    return best, assignment, max_rounds


def qbsolv_multistart(problem, piece_size, restarts=20, max_rounds=20):
    """Budget-equalised rival: run qbsolv_style_solve from `restarts`
    different random starts, keep the best. Returns (score, assignment,
    best_start_seed)."""
    best_score, best_assignment, best_seed = -1, None, None
    for r in range(restarts):
        s, a, _ = qbsolv_style_solve(problem, piece_size, seed=r,
                                     max_rounds=max_rounds)
        if s > best_score:
            best_score, best_assignment, best_seed = s, a, r
    return best_score, best_assignment, best_seed


def greedy_value(problem, v, assignment):
    """Fallback for unembeddable pieces: pick 0/1 for v to disagree
    with as many already-decided neighbours as possible."""
    decided = [assignment[u] for u in problem.neighbors(v) if u in assignment]
    if not decided:
        return 0
    return 1 - round(sum(decided) / len(decided))


def full_pipeline(problem, hardware, cut_knobs, embed_knobs,
                  embed_seed=0, timeout=5):
    """The two workstreams joined: cut -> embed each piece -> solve
    embedded pieces exactly, greedy fallback for failures -> stitch.
    Returns (score, diagnostics). Diagnostics include severed edge count
    (cut quality) and max chain (embedding strain) for the trade story."""
    pieces = cut(problem, cut_knobs["piece_size"], cut_knobs["overlap"],
                 cut_knobs.get("order_seed"),
                 ordering="shuffle" if cut_knobs.get("order_seed") is not None
                 else "index")

    internal = set()
    for piece in pieces:
        pset = set(piece)
        internal.update((u, v) for u, v in problem.edges
                        if u in pset and v in pset)
    severed = problem.number_of_edges() - len(internal)

    assignment, embedded, failed, max_chain = {}, 0, 0, 0
    for piece in pieces:
        sub = problem.subgraph(piece)
        if sub.number_of_edges() == 0:
            emb = {v: [0] for v in piece}      # nothing to wire: trivially fits
        else:
            emb = minorminer.find_embedding(
                list(sub.edges), list(hardware.edges),
                timeout=timeout, random_seed=embed_seed, **embed_knobs)
        if emb:
            embedded += 1
            max_chain = max(max_chain, max(len(c) for c in emb.values()))
            assignment.update(solve_piece(problem, piece, assignment))
        else:
            failed += 1
            for v in piece:
                if v not in assignment:
                    assignment[v] = greedy_value(problem, v, assignment)

    return score_assignment(problem, assignment), {
        "pieces": len(pieces), "severed": severed,
        "embedded": embedded, "failed": failed, "max_chain": max_chain}


import numpy as np
import networkx as nx


def spectral_order(problem):
    """Order variables so strongly-connected ones sit adjacent, using
    the Fiedler vector computed by dense eigendecomposition -- exact
    same ordering as before, but milliseconds at n<=500 and cannot
    hang (the iterative solver stalled on regular graphs at scale)."""
    def _order_component(g):
        nodes = sorted(g.nodes)
        if len(nodes) <= 2:
            return nodes
        L = nx.laplacian_matrix(g, nodelist=nodes).toarray().astype(float)
        _, vecs = np.linalg.eigh(L)
        return [n for _, n in sorted(zip(vecs[:, 1], nodes))]

    if nx.is_connected(problem):
        return _order_component(problem)
    order = []
    for comp in nx.connected_components(problem):
        order += _order_component(problem.subgraph(comp))
    return order


import dimod
from dwave.samplers import SimulatedAnnealingSampler

_sampler = SimulatedAnnealingSampler()


_sampler = SimulatedAnnealingSampler()


def solve_piece_annealed(problem, piece, fixed, embedding,
                         chain_strength=2.0, num_reads=10, seed=0):
    """Solve one piece the way real hardware does: through its embedding.
    Chains are coupled with chain_strength, the embedded object is
    annealed, answers read per qubit, chains resolved by majority vote
    -- so long chains can break and corrupt answers. Returns values
    for the free variables only."""
    free = [v for v in piece if v not in fixed]
    if not free:
        return {}

    h, J = {}, {}
    for v in free:
        h[v] = 0.0
    for u, v in problem.edges:
        if u in free and v in free:
            J[(u, v)] = 1.0
        elif u in free and v in fixed:
            h[u] += 1.0 if fixed[v] else -1.0
        elif v in free and u in fixed:
            h[v] += 1.0 if fixed[u] else -1.0

    qh, qJ = {}, {}
    for v in free:
        chain = embedding[v]
        for q in chain:
            qh[q] = h[v] / len(chain)
        for a, b in zip(chain, chain[1:]):
            qJ[(a, b)] = -chain_strength
    for (u, v), j in J.items():
        cu, cv = embedding[u], embedding[v]
        a = cu[(u + v) % len(cu)]
        b = cv[(u + v) % len(cv)]
        qJ[(a, b)] = qJ.get((a, b), 0.0) + j

    result = _sampler.sample_ising(qh, qJ, num_reads=num_reads, seed=seed)
    best = result.first.sample

    out = {}
    for v in free:
        votes = [best[q] for q in embedding[v]]
        out[v] = 1 if sum(votes) < 0 else 0
    return out


def embed_piece(problem, piece, hardware, embed_knobs, seed=0, timeout=2):
    """Embed one piece's subgraph; returns the embedding dict or {}.
    Variables with no internal edges get trivial one-qubit placements."""
    sub = problem.subgraph(piece)
    if sub.number_of_edges() == 0:
        return {v: [f"t{i}"] for i, v in enumerate(piece)}
    import minorminer
    emb = minorminer.find_embedding(
        list(sub.edges), list(hardware.edges),
        timeout=timeout, random_seed=seed, **embed_knobs)
    if emb:
        spare = 0
        for v in piece:
            if v not in emb:               # isolated within this piece
                emb[v] = [f"iso{spare}"]
                spare += 1
    return emb


def full_pipeline_real(problem, hardware, cut_knobs, embed_knobs,
                       embed_seed=0, timeout=2, chain_strength=1.0):
    """Realistic pipeline: decompose -> embed each piece -> anneal
    THROUGH the embedding (chains can break) -> stitch. Returns
    (score, diagnostics). No penalty needed: bad embeddings hurt
    physically."""
    pieces = cut(problem, cut_knobs["piece_size"], cut_knobs["overlap"],
                 cut_knobs.get("order_seed"),
                 ordering=cut_knobs.get("ordering", "spectral"))
    assignment, failed, max_chain = {}, 0, 0
    for piece in pieces:
        emb = embed_piece(problem, piece, hardware, embed_knobs,
                          seed=embed_seed, timeout=timeout)
        if emb:
            max_chain = max(max_chain,
                            max(len(c) for c in emb.values()))
            assignment.update(solve_piece_annealed(
                problem, piece, assignment, emb,
                chain_strength=chain_strength, seed=embed_seed))
        else:
            failed += 1
            for v in piece:
                if v not in assignment:
                    assignment[v] = greedy_value(problem, v, assignment)
    return score_assignment(problem, assignment), {
        "pieces": len(pieces), "failed": failed, "max_chain": max_chain}