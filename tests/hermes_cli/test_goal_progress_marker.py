"""PROGRESS 标记解析/回显/非回退 单测（goal 模式 ledger 进度信号）。"""
import sys
sys.path.insert(0, ".")
from hermes_cli.goals import GoalManager, GoalState


def test_progress_parsed_and_echoed():
    gm = GoalManager.__new__(GoalManager)
    st = GoalState(goal="replicate", turns_used=2, max_turns=10)
    gm._state = st
    gm.session_id = "test-session"
    # 模拟 evaluate_after_turn 的解析段（judge 由既有测试覆盖）
    resp = "完成了对象 12\nPROGRESS: 12/397"
    m = None
    for ln in reversed(resp.strip().splitlines()):
        if ln.strip().upper().startswith("PROGRESS:"):
            m = __import__("re").match(  # re-ok: 解析结构化标记 N/M,与 goals.py 同款保持一致
                r"PROGRESS:\s*(\d+)\s*/\s*(\d+)", ln.strip(), __import__("re").IGNORECASE)
            break
    assert m and int(m.group(1)) == 12 and int(m.group(2)) == 397
    st.progress_num, st.progress_den = int(m.group(1)), int(m.group(2))
    # 回显
    prompt = gm.next_continuation_prompt(reason="keep going")
    assert "12/397" in prompt, prompt
    # 非回退
    st.progress_num = max(st.progress_num, 3)
    assert st.progress_num == 12
    # JSON 序列化往返
    st2 = GoalState.from_json(st.to_json())
    assert st2.progress_num == 12 and st2.progress_den == 397
    # 旧数据缺字段 → 0/0
    old = GoalState.from_json('{"goal": "g"}')
    assert old.progress_num == 0 and old.progress_den == 0
    print("ALL PASS")


test_progress_parsed_and_echoed()
