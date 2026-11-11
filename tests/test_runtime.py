"""Agent Runtime 主循环：直答 / 工具 / 多轮 / 轮次上限 / 异常 / context 取舍。"""
import json
import unittest

from mini_agent.core.context import ContextPolicy
from mini_agent.core.registry import ToolRegistry
from mini_agent.core.types import ToolSpec
from mini_agent.tools.base import build_minimal_registry

try:
    from .helpers import all_text, call, final_json, make_runtime, tool_json
except ImportError:
    from helpers import all_text, call, final_json, make_runtime, tool_json


def _boom(*args, **kwargs):
    raise RuntimeError("network down")


class TestBasicLoop(unittest.TestCase):
    def test_direct_reply_without_tool(self):
        rt, llm = make_runtime([final_json("你好呀")])
        res = rt.run("s1", "在吗")
        self.assertEqual(res.status, "ok")
        self.assertEqual(res.answer, "你好呀")
        self.assertEqual(res.turns, 1)
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(res.trace.tool_calls(), [])
        self.assertEqual(len(res.trace.of_type("final")), 1)

    def test_tool_then_final(self):
        rt, llm = make_runtime([
            tool_json("需要计算", [call("calculator", expression="1+2")]),
            final_json("结果是 3"),
        ])
        res = rt.run("s1", "算一下 1+2")
        self.assertEqual(res.status, "ok")
        self.assertEqual(res.answer, "结果是 3")
        self.assertEqual(res.turns, 2)
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(len(res.trace.tool_calls()), 1)
        self.assertEqual(len(res.trace.tool_results()), 1)
        self.assertTrue(res.trace.tool_results()[0]["ok"])
        self.assertEqual([e.payload["decision"] for e in res.trace.of_type("decision")],
                         ["tool", "final"])

    def test_multiple_parallel_tool_calls(self):
        rt, _ = make_runtime([
            tool_json("两件事都要做", [
                call("weather", city="广州"),
                call("todo", action="add", item="喷药"),
            ]),
            final_json("都办好了"),
        ])
        res = rt.run("s1", "查天气并记待办")
        self.assertEqual(len(res.trace.tool_calls()), 2)
        self.assertEqual([p["name"] for p in res.trace.tool_calls()], ["weather", "todo"])
        todos = rt.sessions.get("s1").state["todos"]
        self.assertEqual(todos[0]["item"], "喷药")

    def test_tool_result_decides_to_continue_loop(self):
        rt, llm = make_runtime([
            tool_json("先算", [call("calculator", expression="2*3")]),
            tool_json("再查天气", [call("weather", city="北京")]),
            final_json("全部完成"),
        ])
        res = rt.run("s1", "依次做两件事")
        self.assertEqual(res.status, "ok")
        self.assertEqual(res.turns, 3)
        self.assertEqual(len(llm.calls), 3)
        self.assertEqual(len(res.trace.tool_calls()), 2)
        # 第 2 轮的 messages 里必须能看到第 1 轮的工具结果
        self.assertIn("<tool_result", all_text(llm.calls[1]))
        self.assertIn("calculator", all_text(llm.calls[2]))

    def test_tool_schemas_are_handed_to_llm(self):
        captured = {}

        def first(messages, schemas, force_final):
            captured["schemas"] = schemas
            return final_json("ok")

        rt, _ = make_runtime([first])
        rt.run("s1", "hi")
        names = [s["name"] for s in captured["schemas"]]
        self.assertIn("calculator", names)
        self.assertIn("search", names)
        self.assertIn("todo", names)


class TestErrorHandlingInLoop(unittest.TestCase):
    def test_unknown_tool_error_is_fed_back_not_raised(self):
        rt, llm = make_runtime([
            tool_json("试试", [{"name": "not_registered", "arguments": {}}]),
            final_json("已处理工具错误"),
        ])
        res = rt.run("s1", "go")
        self.assertEqual(res.status, "ok")
        self.assertFalse(res.trace.tool_results()[0]["ok"])
        self.assertIn("unknown_tool", all_text(llm.calls[1]))

    def test_invalid_arguments_fed_back(self):
        rt, llm = make_runtime([
            tool_json("漏参数", [{"name": "calculator", "arguments": {}}]),
            final_json("已修正"),
        ])
        res = rt.run("s1", "go")
        self.assertEqual(res.status, "ok")
        self.assertIn("参数校验失败", all_text(llm.calls[1]))

    def test_handler_exception_does_not_break_loop(self):
        reg = build_minimal_registry()

        def boom(args, ctx):
            raise RuntimeError("kaboom")

        reg.register(ToolSpec("boom", "always fails", {"type": "object", "properties": {}}), boom)
        rt, llm = make_runtime([
            tool_json("试试", [call("boom")]),
            final_json("工具坏了但我没崩"),
        ], registry=reg)
        res = rt.run("s1", "go")
        self.assertEqual(res.status, "ok")
        self.assertIn("kaboom", all_text(llm.calls[1]))

    def test_unparsable_output_triggers_repair(self):
        rt, llm = make_runtime([
            json.dumps({"thought": "犹豫中", "action": {}}),
            final_json("修复后的答案"),
        ])
        res = rt.run("s1", "go")
        self.assertEqual(res.status, "ok")
        self.assertEqual(res.answer, "修复后的答案")
        self.assertEqual(len(res.trace.of_type("repair")), 1)
        self.assertEqual(len(llm.calls), 2)
        # REPAIR_HINT 是临时注入的，不写进会话历史
        self.assertNotIn("重新输出", all_text(rt.sessions.get("s1").context.messages))

    def test_llm_failure_returns_structured_error(self):
        rt, _ = make_runtime([_boom, _boom], max_llm_retries=1)
        res = rt.run("s1", "go")
        self.assertEqual(res.status, "error")
        self.assertIn("异常", res.answer)
        self.assertIn("LLM", res.error or "")
        self.assertEqual(len(res.trace.errors()), 1)
        self.assertGreaterEqual(len(res.trace.of_type("warning")), 2)

    def test_invalid_session_id_does_not_raise(self):
        rt, _ = make_runtime([])
        res = rt.run("", "go")
        self.assertEqual(res.status, "error")


class TestLoopLimit(unittest.TestCase):
    def test_max_rounds_forces_final_answer(self):
        rt, llm = make_runtime([
            tool_json("继续", [call("calculator", expression="1+1")]),
            tool_json("还继续", [call("calculator", expression="2+2")]),
            final_json("强制收尾答案"),
        ], max_tool_rounds=2)
        res = rt.run("s1", "一直调工具")
        self.assertEqual(res.status, "loop_limit")
        self.assertEqual(res.answer, "强制收尾答案")
        self.assertEqual(len(res.trace.of_type("loop_limit")), 1)
        self.assertTrue(llm.force_final_flags[-1])
        self.assertIn("强制收尾", llm.calls[2][0].content)
        self.assertEqual(len(res.trace.tool_calls()), 2)

    def test_fallback_when_model_still_wants_tools_at_limit(self):
        rt, _ = make_runtime([
            tool_json("a", [call("calculator", expression="1+1")]),
            tool_json("b", [call("weather", city="北京")]),
            tool_json("c", [call("calculator", expression="3+3")]),
        ], max_tool_rounds=2)
        res = rt.run("s1", "一直调工具")
        self.assertEqual(res.status, "loop_limit")
        self.assertIn("已达到最大工具轮次", res.answer)
        self.assertIn("calculator", res.answer)


class TestContextDecisions(unittest.TestCase):
    def test_thought_excluded_from_llm_context_but_kept_in_audit_and_trace(self):
        rt, llm = make_runtime([
            tool_json("SECRET_THOUGHT_MARKER", [call("calculator", expression="1+1")]),
            final_json("完成", "SECOND_THOUGHT_MARKER"),
        ])
        res = rt.run("s1", "算一下")
        self.assertNotIn("SECRET_THOUGHT_MARKER", all_text(llm.calls[1]))
        self.assertIn("SECRET_THOUGHT_MARKER",
                      [e.payload["thought"] for e in res.trace.of_type("thought")])
        self.assertEqual(rt.sessions.get("s1").context.audit[0]["thought"],
                         "SECRET_THOUGHT_MARKER")

    def test_thought_included_when_policy_enabled(self):
        policy = ContextPolicy(include_thought_in_llm_context=True)
        rt, llm = make_runtime([
            tool_json("INCLUDE_ME", [call("calculator", expression="1+1")]),
            final_json("完成"),
        ], policy=policy)
        rt.run("s1", "算一下")
        self.assertIn("INCLUDE_ME", all_text(llm.calls[1]))

    def test_tool_results_enter_context(self):
        rt, llm = make_runtime([
            tool_json("t", [call("calculator", expression="7*6")]),
            final_json("答案 42"),
        ])
        rt.run("s1", "算一下")
        second_round = llm.calls[1]
        self.assertTrue(any(m.role == "tool" for m in second_round))
        self.assertIn("<tool_result", all_text(second_round))
        self.assertIn("42", all_text(second_round))

    def test_state_snapshot_injected_on_later_run(self):
        rt, llm = make_runtime([
            tool_json("记一下", [call("todo", action="add", item="写周报X")]),
            final_json("已记录"),
            final_json("继续回答"),
        ])
        rt.run("s1", "记个待办")
        rt.run("s1", "继续聊")
        text = all_text(llm.calls[2])
        self.assertIn("Session state snapshot", text)
        self.assertIn("写周报X", text)

    def test_pure_conversation_followup_keeps_history(self):
        rt, llm = make_runtime([final_json("第一轮答案"), final_json("第二轮答案")])
        rt.run("s1", "第一个问题")
        rt.run("s1", "接着刚才说")
        text = all_text(llm.calls[1])
        self.assertIn("第一个问题", text)
        self.assertIn("第一轮答案", text)

    def test_tool_followup_keeps_previous_tool_result(self):
        rt, llm = make_runtime([
            tool_json("t", [call("calculator", expression="3*3")]),
            final_json("结果是 9"),
            final_json("基于刚才结果继续"),
        ])
        rt.run("s1", "算一下")
        rt.run("s1", "基于刚才的结果再说说")
        text = all_text(llm.calls[2])
        self.assertIn("<tool_result", text)
        self.assertIn("结果是 9", text)

    def test_custom_tool_works_without_runtime_changes(self):
        reg = build_minimal_registry()

        @reg.tool("echo", "回显", {"type": "object",
                                   "properties": {"text": {"type": "string"}},
                                   "required": ["text"]})
        def echo(args, ctx):
            return {"text": args["text"]}

        rt, _ = make_runtime([
            tool_json("t", [call("echo", text="hello")]),
            final_json("done"),
        ], registry=reg)
        res = rt.run("s1", "echo")
        self.assertEqual(res.status, "ok")
        self.assertTrue(res.trace.tool_results()[0]["ok"])
        self.assertIn("hello", res.trace.tool_results()[0]["preview"])


if __name__ == "__main__":
    unittest.main()
