"""端到端测试：最少改判方案的状态接口、批量应用、事务与不变量。

python3 tests/test_api.py
"""
import os
import sys
import tempfile
from pathlib import Path

# 导入前指定独立测试数据库
_tmp = tempfile.mkdtemp()
os.environ["DECISION_DB"] = str(Path(_tmp) / "test.db")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from app.main import app, get_conn

client = TestClient(app)


def make_session(cands):
    r = client.post("/api/sessions", json={"candidates": cands})
    assert r.status_code == 201, r.text
    return r.json()["session_id"]


def judge_all(sid, judgments):
    """按顺序判定；judgments 为 (a, b, result)，需与当前待比较组合一致。"""
    for a, b, result in judgments:
        r = client.post(f"/api/sessions/{sid}/judgments", json={"a": a, "b": b, "result": result})
        assert r.status_code == 201, r.text
    return r.json()


def judgment_rows(sid):
    conn = get_conn()
    try:
        return conn.execute(
            "SELECT id, a, b, result FROM judgments WHERE session_id = ? ORDER BY id", (sid,)
        ).fetchall()
    finally:
        conn.close()


def test_fix_suggestion_and_apply():
    # 4 候选，构造 甲→乙→丙→甲 环 + 一条无关边
    sid = make_session(["甲", "乙", "丙", "丁"])
    state = judge_all(sid, [
        ("甲", "乙", "a"),   # 甲 胜 乙
        ("甲", "丙", "b"),   # 丙 胜 甲  → 环 甲→乙→丙→甲
        ("甲", "丁", "a"),   # 甲 胜 丁（不在环上）
        ("乙", "丙", "a"),   # 乙 胜 丙
        ("乙", "丁", "tie"), # 平局不参与计算
        ("丙", "丁", "a"),   # 丙 胜 丁
    ])
    assert state["conflict"] is not None
    fix = state["conflict"]["fix"]
    # 环上 3 条严格边任改一条即可消环；按判断记录顺序应选最早的「甲 胜 乙」
    assert fix["changes"] == 1, fix
    assert fix["pairs"] == [{"a": "甲", "b": "乙", "result": "a"}], fix

    before = judgment_rows(sid)
    r = client.post(f"/api/sessions/{sid}/apply-fix")
    assert r.status_code == 200, r.text
    state = r.json()
    assert state["conflict"] is None, "改判后不应再有冲突"
    assert state["progress"] == {"done": 6, "total": 6}, "进度不变"

    after = judgment_rows(sid)
    assert [row["id"] for row in after] == [row["id"] for row in before], "记录 id 不变"
    assert len(after) == len(before), "记录数不变"
    changed = [(r1["a"], r1["b"], r1["result"], r2["result"])
               for r1, r2 in zip(before, after) if r1["result"] != r2["result"]]
    assert changed == [("甲", "乙", "a", "tie")], f"仅目标组合被改为平局: {changed}"

    # 撤销顺序不变：undo 仍删除 id 最大的记录（丙 胜 丁）
    r = client.post(f"/api/sessions/{sid}/undo")
    assert r.status_code == 200
    rows = judgment_rows(sid)
    assert [row["id"] for row in rows] == [row["id"] for row in before[:-1]]
    assert rows[-1]["a"] == "乙" and rows[-1]["b"] == "丁"
    print("✓ 建议方案 + 一键应用 + id/进度/撤销顺序不变")


def test_apply_fix_no_conflict_409():
    sid = make_session(["A", "B", "C"])
    state = judge_all(sid, [("A", "B", "a"), ("A", "C", "a"), ("B", "C", "a")])
    assert state["conflict"] is None
    assert "fix" not in (state["conflict"] or {}), "无冲突时不返回建议"
    r = client.post(f"/api/sessions/{sid}/apply-fix")
    assert r.status_code == 409
    print("✓ 无冲突时不显示建议且应用返回 409")


def test_multi_cycle_minimal():
    # 两个三角形共享边 丙→甲：甲→乙→丙→甲 与 甲→丁→丙→甲
    sid = make_session(["甲", "乙", "丙", "丁"])
    state = judge_all(sid, [
        ("甲", "乙", "a"),
        ("甲", "丙", "b"),   # 丙 胜 甲（两环共享边）
        ("甲", "丁", "a"),   # 甲 胜 丁
        ("乙", "丙", "a"),
        ("乙", "丁", "tie"),
        ("丙", "丁", "b"),   # 丁 胜 丙
    ])
    fix = state["conflict"]["fix"]
    # 改判共享边「甲 vs 丙」为平局即可同时消除两环
    assert fix["changes"] == 1, fix
    assert fix["pairs"] == [{"a": "甲", "b": "丙", "result": "b"}], fix
    r = client.post(f"/api/sessions/{sid}/apply-fix")
    assert r.status_code == 200
    assert r.json()["conflict"] is None
    print("✓ 多环共享边时改动数最小")


def test_two_disjoint_cycles_need_two():
    # 5 候选，两个边不相交三角形：甲→乙→丙→甲、丙→丁→戊→丙
    sid = make_session(["甲", "乙", "丙", "丁", "戊"])
    state = judge_all(sid, [
        ("甲", "乙", "a"),
        ("甲", "丙", "b"),
        ("甲", "丁", "a"),
        ("甲", "戊", "a"),
        ("乙", "丙", "a"),
        ("乙", "丁", "a"),
        ("乙", "戊", "a"),
        ("丙", "丁", "a"),
        ("丙", "戊", "b"),   # 戊 胜 丙
        ("丁", "戊", "a"),   # 丁 胜 戊 → 环 丙→丁→戊→丙
    ])
    fix = state["conflict"]["fix"]
    assert fix["changes"] == 2, fix
    r = client.post(f"/api/sessions/{sid}/apply-fix")
    assert r.status_code == 200
    assert r.json()["conflict"] is None
    print("✓ 边不相交双环需改判 2 条")


def test_stable_choice_by_record_order():
    # 同一个环 甲→乙→丙→甲，不同录入顺序 → 记录顺序不同 → 稳定选择记录最早者
    # 录入 [甲,乙,丙]：判定顺序 甲乙、甲丙、乙丙 → 应选「甲 胜 乙」
    sid1 = make_session(["甲", "乙", "丙"])
    state = judge_all(sid1, [("甲", "乙", "a"), ("甲", "丙", "b"), ("乙", "丙", "a")])
    fix1 = state["conflict"]["fix"]
    assert (fix1["pairs"][0]["a"], fix1["pairs"][0]["b"]) == ("甲", "乙"), fix1
    # 录入 [乙,丙,甲]：判定顺序 乙丙、乙甲、丙甲 → 应选「乙 胜 丙」
    sid2 = make_session(["乙", "丙", "甲"])
    state = judge_all(sid2, [("乙", "丙", "a"), ("乙", "甲", "b"), ("丙", "甲", "a")])
    fix2 = state["conflict"]["fix"]
    assert (fix2["pairs"][0]["a"], fix2["pairs"][0]["b"]) == ("乙", "丙"), fix2
    print("✓ 多个最优方案按判断记录顺序稳定选择")


def test_partial_judgments_pending_excluded():
    # 未比较组合不参与：严格边构成环，其余 pending 或平局
    sid = make_session(["甲", "乙", "丙", "丁"])
    state = judge_all(sid, [
        ("甲", "乙", "a"),
        ("甲", "丙", "b"),
        ("甲", "丁", "tie"),  # 平局不参与
        ("乙", "丙", "a"),
    ])
    fix = state["conflict"]["fix"]
    assert fix["changes"] == 1
    assert fix["pairs"] == [{"a": "甲", "b": "乙", "result": "a"}]
    r = client.post(f"/api/sessions/{sid}/apply-fix")
    assert r.status_code == 200
    state = r.json()
    assert state["conflict"] is None
    assert state["progress"] == {"done": 4, "total": 6}
    print("✓ 部分判定时平局与未比较组合不参与计算")


def test_ranking_recomputed_after_fix():
    sid = make_session(["甲", "乙", "丙"])
    judge_all(sid, [("甲", "乙", "a"), ("甲", "丙", "b"), ("乙", "丙", "a")])
    r = client.post(f"/api/sessions/{sid}/apply-fix")
    state = r.json()
    # 「甲 胜 乙」改判为平局后：乙 胜 丙、丙 胜 甲、甲 乙 平局 → 乙 > 丙 > 甲
    assert state["ranking"] == [["乙"], ["丙"], ["甲"]], state["ranking"]
    assert state["conflict"] is None
    print("✓ 应用后排名与冲突已重算")


if __name__ == "__main__":
    test_fix_suggestion_and_apply()
    test_apply_fix_no_conflict_409()
    test_multi_cycle_minimal()
    test_two_disjoint_cycles_need_two()
    test_stable_choice_by_record_order()
    test_partial_judgments_pending_excluded()
    test_ranking_recomputed_after_fix()
    print("\n全部端到端测试通过")
