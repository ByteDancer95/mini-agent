"""LLM 输出解析：思考过程 / 工具调用 / 最终答案 的提取与容错。"""
import json
import unittest

from mini_agent.core.parser import OutputParser, parse_llm_output
from mini_agent.core.types import LLMResponse, ToolCall

try:
    from .helpers import call, tool_json
except ImportError:
    from helpers import call, tool_json


class TestParser(unittest.TestCase):
    def setUp(self):
        self.parser = OutputParser()

    # ---- 标准协议 ---------------------------------------------------------
    def test_tool_decision_with_thought(self):
        raw = tool_json("需要算一下", [call("calculator", expression="1+2")])
        out = self.parser.parse(raw)
        self.assertEqual(out.decision(), "tool")
        self.assertEqual(out.thought, "需要算一下")
        self.assertEqual(out.final_answer, None)
        self.assertEqual(len(out.tool_calls), 1)
        self.assertEqual(out.tool_calls[0].name, "calculator")
        self.assertEqual(out.tool_calls[0].arguments, {"expression": "1+2"})
        self.assertEqual(out.source, "json")

    def test_final_answer_with_thought(self):
        raw = json.dumps({"thought": "已有结果", "action": {"type": "final", "content": "答案是 42"}},
                         ensure_ascii=False)
        out = self.parser.parse(raw)
        self.assertTrue(out.is_final)
        self.assertEqual(out.final_answer, "答案是 42")
        self.assertEqual(out.thought, "已有结果")

    def test_multiple_parallel_tool_calls(self):
        raw = tool_json("两个工具互不依赖", [
            call("weather", city="广州"),
            call("todo", action="add", item="喷药"),
        ])
        out = self.parser.parse(raw)
        self.assertEqual([c.name for c in out.tool_calls], ["weather", "todo"])

    # ---- 容错 -------------------------------------------------------------
    def test_markdown_code_fence(self):
        raw = "```json\n" + tool_json("x", [call("search", query="abc")]) + "\n```"
        out = self.parser.parse(raw)
        self.assertEqual(out.tool_calls[0].name, "search")

    def test_json_embedded_in_prose(self):
        raw = "我先说明一下：" + tool_json("x", [call("calculator", expression="2*3")]) + " 就这样。"
        out = self.parser.parse(raw)
        self.assertEqual(out.decision(), "tool")
        self.assertTrue(out.parse_warning)

    def test_flat_tool_calls_shape(self):
        raw = json.dumps({"thought": "t", "tool_calls": [call("weather", city="北京")]})
        out = self.parser.parse(raw)
        self.assertEqual(out.decision(), "tool")
        self.assertEqual(out.tool_calls[0].name, "weather")

    def test_argument_aliases(self):
        variants = [
            {"name": "a", "args": {"x": 1}},
            {"tool": "b", "parameters": {"y": 2}},
            {"name": "c", "arguments": '{"z": 3}'},
            {"function": {"name": "d", "arguments": '{"w": 4}'}},
        ]
        for v in variants:
            raw = json.dumps({"action": {"type": "tool", "tool_calls": [v]}})
            out = self.parser.parse(raw)
            self.assertEqual(len(out.tool_calls), 1, v)
        self.assertEqual(out.tool_calls[0].name, "d")

    def test_final_answer_aliases(self):
        for raw in [
            '{"final_answer": "A"}',
            '{"type": "final", "content": "B"}',
            '{"action": "final", "answer": "C"}',
        ]:
            out = self.parser.parse(raw)
            self.assertTrue(out.is_final, raw)
        self.assertEqual(self.parser.parse('{"final_answer": "A"}').final_answer, "A")

    def test_plain_text_degrades_to_final(self):
        out = self.parser.parse("我不需要工具，直接回答你。")
        self.assertTrue(out.is_final)
        self.assertEqual(out.source, "plain")
        self.assertTrue(out.parse_warning)
        self.assertIn("直接回答", out.final_answer)

    def test_malformed_json_object_is_unusable(self):
        out = self.parser.parse('{"thought": "想了一下", "action": {}}')
        self.assertFalse(out.is_usable)
        self.assertEqual(out.decision(), "invalid")
        self.assertTrue(out.parse_warning)

    def test_empty_output(self):
        out = self.parser.parse("")
        self.assertTrue(out.is_final)
        self.assertIsNotNone(out.final_answer)

    # ---- 原生 function calling --------------------------------------------
    def test_native_tool_calls_take_priority(self):
        resp = LLMResponse(raw='{"action": {"type": "final", "content": "wrong"}}',
                           native_tool_calls=[ToolCall("todo", {"action": "list"})])
        out = self.parser.parse(resp)
        self.assertEqual(out.decision(), "tool")
        self.assertEqual(out.source, "native")
        self.assertEqual(out.tool_calls[0].name, "todo")

    def test_parse_accepts_str_and_response(self):
        self.assertTrue(self.parser.parse('{"final_answer": "x"}').is_final)
        self.assertTrue(self.parser.parse(LLMResponse(raw='{"final_answer":"y"}')).is_final)


if __name__ == "__main__":
    unittest.main()
