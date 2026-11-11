"""真实模型坏输出 + 系统不变量：测「行为」，不测「实现」。

这里的语料不是我构造的完美协议 JSON，而是真实模型常见的翻车输出
（围栏、半截 JSON、全角标点、arguments 被字符串化、幻觉工具名、空输出……）。
断言的不是某个字段在不在，而是系统在任意这类输入下都满足的不变量：
不抛异常、loop 必收敛、context 有界、session 不串台、工具异常不外溢。
"""
from __future__ import annotations

import json
import logging
import random
import unittest
from pathlib import Path

logging.getLogger("mini_agent").setLevel(logging.CRITICAL)

from mini_agent.core.context import ContextPolicy  # noqa: E402
from mini_agent.core.parser import OutputParser  # noqa: E402
from mini_agent.core.types import ToolSpec  # noqa: E402
from mini_agent.tools.base import build_minimal_registry  # noqa: E402

try:
    from .helpers import call, final_json, make_heuristic_runtime, make_runtime, tool_json
except ImportError:
    from helpers import call, final_json, make_heuristic_runtime, make_runtime, tool_json

FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures" / "llm_outputs.json").read_text(encoding="utf-8")
)
BY_ID = {c["id"]: c["raw"] for c in FIXTURES}


class TestMessyRealWorldOutputs(unittest.TestCase):
    def test_every_fixture_converges_without_crashing(self):
        for case in FIXTURES:
            with self.subTest(case=case["id"]):
                rt, _ = make_runtime([case["raw"]], max_tool_rounds=3)
                result = rt.run("s", "随便问一句")
                self.assertNotEqual(result.status, "error", case["id"])
                self.assertTrue(result.answer.strip(), case["id"])
                self.assertLessEqual(result.turns, 4, case["id"])
                self.assertTrue(result.trace.events, case["id"])
                self.assertNotIn("Traceback", result.answer, case["id"])

    def test_parser_behaviour_on_representative_failures(self):
        parser = OutputParser()
        self.assertEqual(parser.parse(BY_ID["fenced_tool_call"]).decision(), "tool")
        self.assertEqual(parser.parse(BY_ID["fenced_tool_call"]).tool_calls[0].name,
                         "calculator")
        plain = parser.parse(BY_ID["plain_natural_language"])
        self.assertTrue(plain.is_final)
        self.assertEqual(plain.source, "plain")
        self.assertTrue(plain.parse_warning)
        truncated = parser.parse(BY_ID["truncated_json"])
        self.assertTrue(truncated.is_final)
        self.assertEqual(truncated.source, "plain")
        with_string_args = parser.parse(BY_ID["arguments_as_json_string"])
        self.assertEqual(with_string_args.tool_calls[0].arguments, {"expression": "2*3"})
        nulls = parser.parse(BY_ID["null_entry_in_tool_calls"])
        self.assertEqual(len(nulls.tool_calls), 1)

    def test_unknown_tool_and_invalid_args_are_recoverable(self):
        for key in ["unknown_tool", "invalid_tool_arguments"]:
            with self.subTest(case=key):
                rt, _ = make_runtime([BY_ID[key]], max_tool_rounds=3)
                result = rt.run("s", "go")
                self.assertNotEqual(result.status, "error")
                self.assertTrue(result.answer.strip())
                self.assertTrue(result.trace.tool_results())
                self.assertFalse(result.trace.tool_results()[0]["ok"])


class TestSystemInvariants(unittest.TestCase):
    def test_random_malformed_streams_always_terminate(self):
        rng = random.Random(20261111)
        pool = [c["raw"] for c in FIXTURES] + [
            "{", "}", "null", "[]", "\x00", "{'x': 1}",
            '{"action":{"type":"tool","tool_calls":[{"name":null}]}}',
        ]
        for trial in range(40):
            responses = [rng.choice(pool) for _ in range(rng.randint(1, 8))]
            rt, _ = make_runtime(responses, max_tool_rounds=4)
            with self.subTest(trial=trial):
                result = rt.run("s", "任意输入")
                self.assertIn(result.status, {"ok", "loop_limit"})
                self.assertTrue(result.answer.strip())
                self.assertLessEqual(result.turns, 5)
                self.assertTrue(result.trace.events)
                self.assertTrue(result.trace.of_type("final"))

    def test_tool_exception_never_escapes_the_loop(self):
        reg = build_minimal_registry()

        def flaky(args, ctx):
            raise RuntimeError("upstream tool blew up")

        reg.register(ToolSpec("flaky", "随机失败的工具",
                              {"type": "object", "properties": {}}), flaky)
        responses = [tool_json("t", [call("flaky")]) for _ in range(3)]
        rt, _ = make_runtime(responses, registry=reg, max_tool_rounds=3)
        result = rt.run("s", "go")
        self.assertNotEqual(result.status, "error")
        self.assertTrue(result.answer.strip())
        self.assertGreaterEqual(len(result.trace.tool_results()), 1)
        self.assertTrue(all(not e["ok"] for e in result.trace.tool_results()))
        self.assertEqual(len(result.trace.errors()), 0)

    def test_context_stays_bounded_across_long_pure_conversation(self):
        """纯对话（不调工具）也必须压缩 —— 曾经只在工具分支压缩，这是修掉的 bug。"""
        policy = ContextPolicy(max_messages=8, max_chars=1500, keep_recent=4,
                               compress=True, digest_max_lines=10)
        rt, _ = make_runtime([final_json(f"第 {i} 轮回答") for i in range(60)], policy=policy)
        for i in range(60):
            self.assertNotEqual(rt.run("long", f"问题{i}").status, "error")
        ctx = rt.sessions.get_or_create("long").context
        # 压缩按「超过 max_messages 才触发」，所以稳态上界是 max_messages + 一次 run 的增量，
        # 而不是无界的 60 轮 × 2 条消息
        self.assertLessEqual(len(ctx.messages), policy.max_messages + 2)
        self.assertLess(len(ctx.messages), 20, "纯对话长会话不应线性增长")
        self.assertGreater(ctx.stats()["compactions"], 0)
        self.assertTrue(any(m.role == "user" and "问题59" in m.content
                            for m in ctx.messages), "压缩不能丢掉最近一轮用户输入")

    def test_two_sessions_never_share_state_under_random_interleaving(self):
        rt, _ = make_heuristic_runtime()
        rng = random.Random(7)
        for i in range(30):
            sid = rng.choice(["A", "B"])
            rt.run(sid, f"记个待办：{sid}-任务{i}")
        sa = rt.sessions.get_or_create("A")
        sb = rt.sessions.get_or_create("B")
        self.assertTrue(sa.state["todos"] and sb.state["todos"])
        self.assertTrue(all(t["item"].startswith("A-") for t in sa.state["todos"]))
        self.assertTrue(all(t["item"].startswith("B-") for t in sb.state["todos"]))
        text_a = "\n".join(m.content for m in sa.context.messages)
        text_b = "\n".join(m.content for m in sb.context.messages)
        self.assertNotIn("B-任务", text_a)
        self.assertNotIn("A-任务", text_b)


if __name__ == "__main__":
    unittest.main()
