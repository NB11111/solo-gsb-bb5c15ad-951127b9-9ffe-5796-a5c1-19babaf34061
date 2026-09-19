"""fixplan.min_tie_fix 正确性测试：与暴力枚举对比。

python3 tests/test_fixplan.py
"""
import itertools
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.fixplan import min_tie_fix


def brute_force(candidates, judgments):
    """枚举所有改判子集，返回 (最小改动数, 按记录顺序字典序最小的组合序列)。"""
    strict = [j for j in judgments if j["result"] != "tie"]
    m = len(strict)

    def acyclic(mask):
        # mask: 被改判（删除）的边集合
        adj = {c: set() for c in candidates}
        for i, j in enumerate(strict):
            if mask >> i & 1:
                continue
            u, v = (j["a"], j["b"]) if j["result"] == "a" else (j["b"], j["a"])
            adj[u].add(v)
        # Kahn 拓扑排序
        indeg = {c: 0 for c in candidates}
        for u in adj:
            for v in adj[u]:
                indeg[v] += 1
        stack = [c for c in candidates if indeg[c] == 0]
        seen = 0
        while stack:
            u = stack.pop()
            seen += 1
            for v in adj[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    stack.append(v)
        return seen == len(candidates)

    best_k, best_seq = None, None
    for k in range(m + 1):
        for combo in itertools.combinations(range(m), k):
            mask = sum(1 << i for i in combo)
            if acyclic(mask):
                seq = [(strict[i]["a"], strict[i]["b"]) for i in combo]
                return k, seq
    return 0, []


def check(candidates, judgments):
    k, pairs = min_tie_fix(candidates, judgments)
    bk, bseq = brute_force(candidates, judgments)
    got_seq = [(p["a"], p["b"]) for p in pairs]
    assert k == bk, f"改动数不符: got {k}, want {bk}; judgments={judgments}"
    assert k == len(pairs), f"改动数与方案长度不一致: {k} vs {len(pairs)}"
    assert got_seq == bseq, f"方案序列不符: got {got_seq}, want {bseq}; judgments={judgments}"
    # 方案确实消环
    remaining = [
        j for j in judgments
        if j["result"] != "tie" and (j["a"], j["b"]) not in set(got_seq)
    ]
    bk2, _ = brute_force(candidates, remaining)
    assert bk2 == 0, "应用方案后仍有环"


def main():
    rng = random.Random(20260919)

    # 固定用例：经典 3 环
    check(["甲", "乙", "丙"], [
        {"a": "甲", "b": "乙", "result": "a"},
        {"a": "甲", "b": "丙", "result": "b"},
        {"a": "乙", "b": "丙", "result": "a"},
    ])
    # 平局与未比较不参与
    check(["甲", "乙", "丙", "丁"], [
        {"a": "甲", "b": "乙", "result": "tie"},
        {"a": "甲", "b": "丙", "result": "a"},
        {"a": "甲", "b": "丁", "result": "b"},
        {"a": "乙", "b": "丙", "result": "a"},
        {"a": "乙", "b": "丁", "result": "b"},
        {"a": "丙", "b": "丁", "result": "a"},
    ])
    # 无环
    check(["A", "B", "C"], [
        {"a": "A", "b": "B", "result": "a"},
        {"a": "A", "b": "C", "result": "a"},
        {"a": "B", "b": "C", "result": "a"},
    ])

    # 随机用例：4~8 个候选，随机判定（含平局与未比较）
    for trial in range(300):
        n = rng.randint(3, 8)
        candidates = [f"c{i}" for i in range(n)]
        judgments = []
        for a, b in itertools.combinations(candidates, 2):
            r = rng.random()
            if r < 0.15:
                continue  # 未比较
            result = rng.choice(["a", "b", "tie"]) if r < 0.35 else rng.choice(["a", "b"])
            judgments.append({"a": a, "b": b, "result": result})
        rng.shuffle(judgments)  # 判断记录顺序随机
        check(candidates, judgments)

    print("全部通过：300 组随机用例 + 3 组固定用例")


if __name__ == "__main__":
    main()
