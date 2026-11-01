"""Max-flow / min-cut from scratch (Dinic's algorithm).

The assignment suggests Edmonds-Karp; we implement Dinic's instead, which is
the same augmenting-path idea organised better. Edmonds-Karp augments along one
shortest path per BFS (O(V*E) augmentations, O(V*E^2) overall). Dinic's groups
all shortest augmenting paths of a given length into one "phase": a BFS builds a
level graph, then a single DFS pass saturates a *blocking flow* within it. The
shortest s-t distance strictly increases each phase, so there are at most V
phases, giving O(V^2 * E). On our region graphs (~200 nodes) this converges in
roughly 10-20 phases instead of several hundred individual augmentations, which
is what keeps the cut inside the time budget.

Capacities are integers (callers pre-scale and round). Integer arithmetic makes
termination exact -- with floats, repeated subtraction of near-zero residuals
can stall the algorithm.
"""


class MaxFlow:
    """Directed graph with integer capacities, supporting max-flow and min-cut.

    Edges are kept in flat lists; edge ``e`` and its residual partner ``e ^ 1``
    are always allocated as a pair, so the reverse edge is found by XOR with 1
    rather than by a lookup.
    """

    __slots__ = ("n", "to", "cap", "g", "_level")

    def __init__(self, n):
        self.n = n
        self.to = []
        self.cap = []
        self.g = [[] for _ in range(n)]

    def add_edge(self, u, v, cap, cap_rev=0):
        """Add u->v with capacity `cap`, and v->u with `cap_rev`."""
        self.g[u].append(len(self.to))
        self.to.append(v)
        self.cap.append(cap)
        self.g[v].append(len(self.to))
        self.to.append(u)
        self.cap.append(cap_rev)

    def _levels(self, s, t):
        """BFS over residual edges -> level graph. False if t unreachable."""
        level = [-1] * self.n
        level[s] = 0
        queue = [s]
        to, cap, g = self.to, self.cap, self.g
        qi = 0
        while qi < len(queue):
            v = queue[qi]
            qi += 1
            lv = level[v] + 1
            for e in g[v]:
                if cap[e] > 0:
                    u = to[e]
                    if level[u] < 0:
                        level[u] = lv
                        queue.append(u)
        self._level = level
        return level[t] >= 0

    def _blocking_flow(self, s, t):
        """Saturate all shortest s->t paths in the current level graph."""
        to, cap, g, level = self.to, self.cap, self.g, self._level
        it = [0] * self.n          # next unexplored edge index per node
        flow = 0
        path = []                  # edge indices of the current partial path
        v = s
        while True:
            if v == t:
                # Augment by the bottleneck along the path.
                f = min(cap[e] for e in path)
                for e in path:
                    cap[e] -= f
                    cap[e ^ 1] += f
                flow += f
                # Retreat to just before the first edge we saturated; every
                # edge upstream of it still has residual capacity.
                cut = 0
                while cut < len(path) and cap[path[cut]] > 0:
                    cut += 1
                del path[cut:]
                # `to[e]` is the head of edge e, so the node we resume from is
                # the head of the last edge still on the path (not its tail).
                v = s if not path else to[path[-1]]
                continue
            # Try to advance along an admissible edge.
            advanced = False
            gv = g[v]
            while it[v] < len(gv):
                e = gv[it[v]]
                u = to[e]
                if cap[e] > 0 and level[u] == level[v] + 1:
                    path.append(e)
                    v = u
                    advanced = True
                    break
                it[v] += 1
            if advanced:
                continue
            # Dead end: remove v from the level graph and back up.
            if v == s:
                break
            level[v] = -1
            e = path.pop()
            it[to[e ^ 1]] += 1
            v = to[e ^ 1]
        return flow

    def max_flow(self, s, t):
        total = 0
        while self._levels(s, t):
            total += self._blocking_flow(s, t)
        return total

    def min_cut_source_side(self, s):
        """Nodes reachable from s in the residual graph after max_flow().

        By the max-flow/min-cut theorem these are exactly the nodes on the
        source side of a minimum cut, i.e. the regions labelled object.
        """
        seen = bytearray(self.n)
        seen[s] = 1
        queue = [s]
        to, cap, g = self.to, self.cap, self.g
        qi = 0
        while qi < len(queue):
            v = queue[qi]
            qi += 1
            for e in g[v]:
                if cap[e] > 0:
                    u = to[e]
                    if not seen[u]:
                        seen[u] = 1
                        queue.append(u)
        return seen
