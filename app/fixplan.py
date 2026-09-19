"""最少改判方案：把最少数量的严格胜负组合改判为平局，消除全部有向环。

仅严格胜负判断构成有向边（胜者 → 败者）；平局与未比较的组合不成边、
不参与计算。把一条边改判为平局等价于从图中删除该边，因此问题即
有向图的最小反馈弧集（minimum feedback arc set）。

候选项最多 12 个：先用子集 DP（Held-Karp 风格）求最大无环子图边数，
得到最小改判数；再按判断记录顺序贪心构造具体方案——逐条尝试将边
纳入改判集，若剩余图仍能在剩余预算内消环则确定纳入。所得方案按
判断记录顺序排列的组合序列字典序最小，多个最优方案间选择稳定，
且只依赖判断记录顺序。
"""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=None)
def _subset_bits(n: int) -> tuple[tuple[int, ...], ...]:
    """每个子集的置位下标（只依赖 n，全局缓存一份）。"""
    out = []
    for s in range(1 << n):
        bits = []
        m = s
        while m:
            lsb = m & -m
            bits.append(lsb.bit_length() - 1)
            m ^= lsb
        out.append(tuple(bits))
    return tuple(out)


def _max_acyclic_edges(in_mask: list[int]) -> int:
    """子集 DP：最大无环子图边数（所有拓扑序下「正向边」数的最大值）。"""
    n = len(in_mask)
    bits = _subset_bits(n)
    dp = [0] * (1 << n)
    for s in range(1, 1 << n):
        best = 0
        for v in bits[s]:
            t = s ^ (1 << v)
            kept = dp[t] + (in_mask[v] & t).bit_count()
            if kept > best:
                best = kept
        dp[s] = best
    return dp[(1 << n) - 1]


def _reachable(in_mask: list[int], src: int, dst: int) -> bool:
    """在 in_mask 表示的有向图中，src 是否能沿边到达 dst。"""
    seen = 1 << src
    changed = True
    while changed:
        changed = False
        for w in range(len(in_mask)):
            if not (seen >> w) & 1 and in_mask[w] & seen:
                seen |= 1 << w
                changed = True
    return bool((seen >> dst) & 1)


def _out_masks(in_mask: list[int]) -> list[int]:
    """由入边掩码算出出边掩码。"""
    n = len(in_mask)
    out = [0] * n
    for v in range(n):
        m = in_mask[v]
        while m:
            lsb = m & -m
            out[lsb.bit_length() - 1] |= 1 << v
            m ^= lsb
    return out


def _greedy_fas_ub(in_mask: list[int]) -> int:
    """启发式上界：按「出度 − 入度」构造初始拓扑序，再做插入式局部搜索，
    返回该序下的反向边数（一个可行改判集的大小，越小越好）。"""
    n = len(in_mask)
    out = _out_masks(in_mask)
    remaining = (1 << n) - 1
    order: list[int] = []
    while remaining:
        best_u, best_d = -1, -(n + 1)
        m = remaining
        while m:
            lsb = m & -m
            u = lsb.bit_length() - 1
            d = (out[u] & remaining).bit_count() - (in_mask[u] & remaining).bit_count()
            if d > best_d:
                best_u, best_d = u, d
            m ^= lsb
        order.append(best_u)
        remaining ^= 1 << best_u

    # 初始反向边数
    pos = [0] * n
    for i, u in enumerate(order):
        pos[u] = i
    best = 0
    for v in range(n):
        m = in_mask[v]
        while m:
            lsb = m & -m
            if pos[lsb.bit_length() - 1] > pos[v]:
                best += 1
            m ^= lsb

    # 插入式局部搜索：相邻交换只影响被越过顶点关联的边，O(1) 增量更新
    improved = True
    while improved:
        improved = False
        for u in range(n):
            i = order.index(u)
            out_u, in_u = out[u], in_mask[u]
            best_pos, cur = i, best
            val = best
            for j in range(i + 1, n):  # u 右移越过 w：u→w 变反向 +1，w→u 变正向 −1
                w = order[j]
                val += ((out_u >> w) & 1) - ((in_u >> w) & 1)
                if val < cur:
                    cur, best_pos = val, j
            val = best
            for j in range(i - 1, -1, -1):  # u 左移越过 w：w→u 变反向 +1，u→w 变正向 −1
                w = order[j]
                val += ((in_u >> w) & 1) - ((out_u >> w) & 1)
                if val < cur:
                    cur, best_pos = val, j
            if best_pos != i:
                order.insert(best_pos, order.pop(i))
                best = cur
                improved = True
    return best


def _shortest_cycle_edges(in_mask: list[int]) -> list[tuple[int, int]] | None:
    """BFS 找一条最短有向环的边列表（用于环包装下界），无环返回 None。"""
    n = len(in_mask)
    out = _out_masks(in_mask)
    best: list[tuple[int, int]] | None = None
    for s in range(n):
        if best is not None and len(best) <= 3:
            break  # 正常数据无自环与 2-环，三角形即理论最短
        parent = [-2] * n
        parent[s] = -1
        depth = [-1] * n
        depth[s] = 0
        frontier = [s]
        found = -1
        while frontier and found < 0:
            nxt = []
            for u in frontier:
                limit = (len(best) - 1) if best else n
                if depth[u] + 1 > limit:
                    continue
                m = out[u]
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    m ^= lsb
                    if v == s:
                        found = u
                        break
                    if parent[v] == -2:
                        parent[v] = u
                        depth[v] = depth[u] + 1
                        nxt.append(v)
                if found >= 0:
                    break
            frontier = nxt
        if found < 0:
            continue
        # 重建 s → … → found → s
        path = [found]
        x = found
        while x != s:
            x = parent[x]
            path.append(x)
        path.reverse()
        cyc = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        cyc.append((found, s))
        if best is None or len(cyc) < len(best):
            best = cyc
    return best


def _has_k_disjoint_cycles(in_mask: list[int], k: int) -> bool:
    """是否存在 k 条边不相交的有向环（若是，则 minFAS ≥ k，为可靠下界）。

    每轮取当前剩余边中的最短环，包装效果优于任意环。
    """
    g = list(in_mask)
    for _ in range(k):
        cycle = _shortest_cycle_edges(g)
        if cycle is None:
            return False
        for u, v in cycle:
            g[v] ^= 1 << u
    return True


def min_tie_fix(candidates: list[str], judgments: list[dict]) -> tuple[int, list[dict]]:
    """计算最少改判方案。

    返回 (改动数, 需改判为平局的组合列表)。组合按判断记录顺序排列，
    每项形如 {"a", "b", "result"}（result 为该组合的当前判定结果）。
    无环时改动数为 0、列表为空。
    """
    idx = {name: i for i, name in enumerate(candidates)}
    # 严格胜负边，按判断记录顺序编号（平局记录跳过、不成边）
    edges: list[tuple[int, int, dict]] = []
    for j in judgments:
        if j["result"] == "tie":
            continue
        a, b = idx[j["a"]], idx[j["b"]]
        u, v = (a, b) if j["result"] == "a" else (b, a)
        edges.append((u, v, j))

    m = len(edges)
    if m == 0:
        return 0, []

    in_mask = [0] * len(candidates)
    for u, v, _ in edges:
        in_mask[v] |= 1 << u

    budget = m - _max_acyclic_edges(in_mask)  # 最小改判数
    if budget == 0:
        return 0, []

    # 贪心：按判断记录顺序，能改判则改判，得到组合序列字典序最小的最优方案。
    # 试改判一条边后需判定「剩余图能否在剩余预算内消环」：先用两个可靠
    # 启发式快速定论（上界：贪心构造的可行改判集；下界：边不相交环包装），
    # 都无法定论时才跑精确的子集 DP。
    chosen: list[dict] = []
    removed = 0
    for u, v, j in edges:
        if budget == 0:
            break
        bit = 1 << u
        if not in_mask[v] & bit:
            continue  # 已改判（防御性判断，正常每组合仅一条记录）
        if not _reachable(in_mask, v, u):
            continue  # 不在任何环上，改判无意义
        in_mask[v] ^= bit  # 试改判（删边）
        removed += 1
        if _greedy_fas_ub(in_mask) <= budget - 1:
            feasible = True
        elif _has_k_disjoint_cycles(in_mask, budget):
            feasible = False
        else:
            feasible = (m - removed) - _max_acyclic_edges(in_mask) <= budget - 1
        if feasible:
            budget -= 1
            chosen.append({"a": j["a"], "b": j["b"], "result": j["result"]})
        else:
            in_mask[v] |= bit  # 恢复
            removed -= 1
    return len(chosen), chosen
