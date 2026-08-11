import networkx as nx
import minorminer
import dwave_networkx as dnx


def find_embedding(problem, hardware):
    """Input: two graphs. Output: embedding {variable: [qubits]} or None."""
    embedding = minorminer.find_embedding(problem.edges, hardware.edges,
                                          random_seed=0)
    return embedding if embedding else None


def verify_embedding(embedding, problem, hardware):
    """Check the three rules that define a valid minor embedding."""
    # Rule 1: every chain is a connected piece of the hardware
    for var, chain in embedding.items():
        if not nx.is_connected(hardware.subgraph(chain)):
            return False, f"chain for {var} is not connected"

    # Rule 2: no physical qubit is used by two different variables
    used = [q for chain in embedding.values() for q in chain]
    if len(used) != len(set(used)):
        return False, "a qubit is shared between two chains"

    # Rule 3: every problem edge has at least one real coupler between chains
    for u, v in problem.edges:
        couplers = [(a, b) for a in embedding[u] for b in embedding[v]
                    if hardware.has_edge(a, b)]
        if not couplers:
            return False, f"no coupler carries problem edge {u}-{v}"

    return True, "valid"


def run_test(name, problem, hardware):
    print("=" * 60)
    print(f"TEST: {name}")
    print(f"  problem : {problem.number_of_nodes()} variables, "
          f"max degree {max(dict(problem.degree).values())}")
    print(f"  hardware: {hardware.number_of_nodes()} qubits, "
          f"max degree {max(dict(hardware.degree).values())}")

    embedding = find_embedding(problem, hardware)
    if embedding is None:
        print("  RESULT  : no embedding found")
        return

    ok, reason = verify_embedding(embedding, problem, hardware)
    print(f"  VERIFIED: {reason}")
    for var in sorted(embedding, key=str):
        chain = embedding[var]
        note = f"  (chain of {len(chain)})" if len(chain) > 1 else ""
        print(f"    {var} -> qubits {sorted(chain)}{note}")
    qubits_used = sum(len(c) for c in embedding.values())
    print(f"  COST    : {qubits_used} physical qubits "
          f"for {problem.number_of_nodes()} variables")


# Test 1: triangle onto a 5 qubit ring
run_test("triangle K3 onto 5 qubit ring",
         problem=nx.complete_graph(3),
         hardware=nx.cycle_graph(5))

# Test 2: star, centre connected to 4 leaves, onto a tree with max degree 3
star4 = nx.star_graph(4)                      # centre 0, leaves 1..4
tree = nx.Graph([("q1", "q2"), ("q1", "q3"), ("q1", "q4"),
                 ("q2", "q5"), ("q2", "q6")])
run_test("star K1,4 (degree 4) onto tree (degree 3)",
         problem=star4, hardware=tree)

# Test 3: star, centre connected to 5 leaves, onto a Chimera unit cell
run_test("star K1,5 (degree 5) onto Chimera unit cell (degree 4)",
         problem=nx.star_graph(5),
         hardware=dnx.chimera_graph(1))