"""Harness 评测闭环的验证：case 驱动、失败判定、机器可读结果、确定性重放。

这是批量评测的驱动模块，**不是新的运行入口**：
    python -m unittest tests.test_harness_eval -v
"""
from __future__ import annotations

import json
import logging
import unittest
from pathlib import Path

from mini_agent.core.eval import (
    EvalCase,
    RecordingLLM,
    load_cases,
    record_and_replay,
    run_case,
    run_suite,
)
from mini_agent.core.runtime import build_default_runtime

logging.getLogger("mini_agent").setLevel(logging.CRITICAL)

ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "tests" / "harness_cases.json"


def runtime_factory():
    """每个 case 一份全新 runtime，保证 session 状态不被别的 case 污染。"""
    return build_default_runtime(llm_kind="heuristic")


class TestHarnessEval(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = load_cases(CASES_PATH)

    def test_case_file_is_well_formed(self) -> None:
        self.assertGreaterEqual(len(self.cases), 5)
        ids = [c.id for c in self.cases]
        self.assertEqual(len(ids), len(set(ids)), "case id 必须唯一")
        self.assertTrue(all(c.turns for c in self.cases))
        self.assertTrue(all(c.session_id for c in self.cases))

    def test_all_builtin_cases_pass(self) -> None:
        suite = run_suite(self.cases, runtime_factory)
        self.assertTrue(
            suite.all_passed,
            json.dumps(suite.to_dict(), ensure_ascii=False, indent=2, default=str),
        )

    def test_suite_result_is_machine_readable(self) -> None:
        suite = run_suite(self.cases[:2], runtime_factory)
        data = json.loads(json.dumps(suite.to_dict(), ensure_ascii=False, default=str))
        self.assertEqual(data["total"], 2)
        self.assertEqual(data["passed"], 2)
        self.assertTrue(data["all_passed"])
        for item in data["results"]:
            self.assertEqual(len(item["trace_summaries"]), len(item["statuses"]))
            self.assertIn("tool_calls", item["trace_summaries"][0])

    def test_wrong_expectation_is_reported_not_raised(self) -> None:
        bad = EvalCase(id="bad", session_id="eval/bad",
                       turns=["帮我算 1+1"], expect_tools=["weather"])
        result = run_case(runtime_factory(), bad)
        self.assertFalse(result.passed)
        self.assertEqual(result.tool_sequence, ["calculator"])
        self.assertTrue(any("工具序列" in f for f in result.failures))

    def test_forbidden_context_detects_cross_session_leak(self) -> None:
        bad = EvalCase(id="leak", session_id="eval/leak",
                       turns=["你好，这是一段独特文本 ABCXYZ"],
                       forbidden_context=["ABCXYZ"])
        result = run_case(runtime_factory(), bad)
        self.assertFalse(result.passed)
        self.assertTrue(any("串入" in f for f in result.failures))

    def test_each_case_replays_deterministically(self) -> None:
        for case in self.cases:
            with self.subTest(case=case.id):
                outcome = record_and_replay(case, runtime_factory)
                self.assertGreater(outcome["recorded_responses"], 0)
                self.assertTrue(
                    outcome["match"],
                    json.dumps(outcome, ensure_ascii=False, indent=2, default=str),
                )

    def test_recorded_responses_are_json_serializable(self) -> None:
        """录制内容能落盘，是「离线复现」与「回归语料」的前提。"""
        runtime = runtime_factory()
        recorder = RecordingLLM(runtime.llm)
        runtime.llm = recorder
        runtime.protocol = recorder.protocol_name
        for text in self.cases[0].turns:
            runtime.run(self.cases[0].session_id, text)
        decoded = json.loads(json.dumps(recorder.recorded, ensure_ascii=False))
        self.assertGreater(len(decoded), 0)
        self.assertIn("raw", decoded[0])
        self.assertIn("native_tool_calls", decoded[0])

    def test_cases_are_isolated_from_each_other(self) -> None:
        """同一 suite 内前序 case 的 todo 不能出现在后序 case 的 session 里。"""
        suite = run_suite(self.cases, runtime_factory)
        by_id = {r.case_id: r for r in suite.results}
        self.assertEqual(by_id["calculator_then_todo_followup"].todos, ["明天交报告"])
        self.assertEqual(by_id["window1_isolated_from_window2"].todos,
                         ["明天给大棚番茄喷药"])


if __name__ == "__main__":
    unittest.main()
