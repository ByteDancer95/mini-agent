"""5 个工具的行为测试，重点：calculator 安全性、todo 会话级隔离。"""
import unittest
from pathlib import Path

from mini_agent.core.types import ToolCall, ToolContext
from mini_agent.tools.base import build_default_registry
from mini_agent.tools.calculator import safe_eval
from mini_agent.tools.read_docs import default_docs_root

try:
    from .helpers import ROOT
except ImportError:
    from helpers import ROOT


def run(name, **kwargs):
    return build_default_registry().execute(ToolCall(name, kwargs), ToolContext("session-x"))


class TestCalculator(unittest.TestCase):
    def test_basic_and_precedence(self):
        self.assertEqual(safe_eval("1+2*3"), 7)
        self.assertEqual(safe_eval("(1+2)*3"), 9)
        self.assertEqual(safe_eval("10/4"), 2.5)
        self.assertEqual(safe_eval("2**8"), 256)

    def test_functions_and_constants(self):
        self.assertAlmostEqual(safe_eval("sqrt(144)/2"), 6.0)
        self.assertEqual(safe_eval("max(3, 9, 2)"), 9)
        self.assertAlmostEqual(safe_eval("round(pi, 2)"), 3.14)

    def test_alias_operators(self):
        self.assertEqual(safe_eval("2^3"), 8)
        self.assertEqual(safe_eval("6×7"), 42)
        self.assertEqual(safe_eval("8÷2"), 4)
        self.assertEqual(safe_eval("3乘以4"), 12)

    def test_division_by_zero_is_error_result(self):
        res = run("calculator", expression="1/0")
        self.assertFalse(res.ok)
        self.assertIn("除数为 0", res.error)

    def test_code_injection_blocked(self):
        """严禁 eval：任意代码 / 属性访问 / 未授权名称一律拒绝。"""
        for expr in [
            "__import__('os').system('echo pwned')",
            "os.system('echo pwned')",
            "open('/etc/passwd').read()",
            "unknown_var + 1",
            "1 if True else 2",
            "[x for x in range(3)]",
        ]:
            with self.subTest(expr=expr):
                res = run("calculator", expression=expr)
                self.assertFalse(res.ok, expr)

    def test_too_long_expression_rejected(self):
        res = run("calculator", expression="1+" * 300 + "1")
        self.assertFalse(res.ok)

    def test_tool_result_shape(self):
        res = run("calculator", expression="18*24")
        self.assertTrue(res.ok)
        self.assertEqual(res.data["result"], 432)
        self.assertEqual(res.data["expression"], "18*24")


class TestSearch(unittest.TestCase):
    def test_mock_results_are_deterministic(self):
        a = run("search", query="番茄 早疫病 防治", top_k=2)
        b = run("search", query="番茄 早疫病 防治", top_k=2)
        self.assertTrue(a.ok)
        self.assertEqual(a.data["results"], b.data["results"])
        self.assertTrue(a.data["mock"])
        self.assertEqual(len(a.data["results"]), 2)

    def test_relevant_doc_ranked_first(self):
        res = run("search", query="周报 模板")
        self.assertTrue(any("周报" in r["title"] for r in res.data["results"]))

    def test_empty_query_rejected_by_schema(self):
        res = run("search", query="")
        self.assertFalse(res.ok)  # minLength=1


class TestTodo(unittest.TestCase):
    def test_full_lifecycle(self):
        state = {}
        ctx = ToolContext("s1", state)

        def call(action, **kw):
            from mini_agent.core.types import ToolCall
            return build_default_registry().execute(
                ToolCall("todo", {"action": action, **kw}), ctx)

        self.assertEqual(len(call("list").data["todos"]), 0)
        added = call("add", item="给番茄喷药", priority="high")
        self.assertTrue(added.ok)
        self.assertEqual(added.data["todos"][0]["id"], 1)
        self.assertEqual(added.data["todos"][0]["priority"], "high")
        done = call("complete", id=1)
        self.assertTrue(done.data["todos"][0]["done"])
        removed = call("remove", id=1)
        self.assertEqual(len(removed.data["todos"]), 0)
        call("add", item="A")
        call("add", item="B")
        self.assertEqual(len(call("clear").data["todos"]), 0)

    def test_add_requires_item(self):
        res = run("todo", action="add")
        self.assertFalse(res.ok)
        self.assertIn("item", res.error)

    def test_state_is_session_scoped(self):
        """两个不同 session 的 state 互不影响（题目核心隔离点）。"""
        from mini_agent.core.types import ToolCall
        reg = build_default_registry()
        s1 = ToolContext("s1", {})
        s2 = ToolContext("s2", {})
        reg.execute(ToolCall("todo", {"action": "add", "item": "窗口1的任务"}), s1)
        reg.execute(ToolCall("todo", {"action": "add", "item": "窗口2的任务"}), s2)
        r1 = reg.execute(ToolCall("todo", {"action": "list"}), s1)
        r2 = reg.execute(ToolCall("todo", {"action": "list"}), s2)
        self.assertEqual([t["item"] for t in r1.data["todos"]], ["窗口1的任务"])
        self.assertEqual([t["item"] for t in r2.data["todos"]], ["窗口2的任务"])
        self.assertEqual(r1.data["session_id"], "s1")
        self.assertEqual(r2.data["session_id"], "s2")


class TestWeather(unittest.TestCase):
    def test_mock_weather(self):
        res = run("weather", city="广州")
        self.assertTrue(res.ok)
        self.assertEqual(res.data["city"], "广州")
        self.assertTrue(res.data["mock"])
        self.assertIn("advice", res.data)

    def test_unknown_city_falls_back(self):
        res = run("weather", city="火星")
        self.assertTrue(res.ok)


class TestReadDocs(unittest.TestCase):
    def test_read_existing_doc(self):
        from mini_agent.core.types import ToolCall
        ctx = ToolContext("s", {}, docs_root=str(default_docs_root()))
        res = build_default_registry().execute(
            ToolCall("read_docs", {"path": "weekly_report_template.md"}), ctx)
        self.assertTrue(res.ok)
        self.assertIn("三段式", res.data["content"])

    def test_path_traversal_blocked(self):
        from mini_agent.core.types import ToolCall
        ctx = ToolContext("s", {}, docs_root=str(default_docs_root()))
        for path in ["../mini_agent/core/runtime.py", "..\\..\\secret.md", "/etc/passwd"]:
            with self.subTest(path=path):
                res = build_default_registry().execute(
                    ToolCall("read_docs", {"path": path}), ctx)
                self.assertFalse(res.ok, path)

    def test_missing_file_and_bad_suffix(self):
        from mini_agent.core.types import ToolCall
        ctx = ToolContext("s", {}, docs_root=str(default_docs_root()))
        reg = build_default_registry()
        self.assertFalse(reg.execute(ToolCall("read_docs", {"path": "nope.md"}), ctx).ok)
        self.assertFalse(reg.execute(ToolCall("read_docs", {"path": "*.py"}), ctx).ok)

    def test_default_docs_root_exists(self):
        self.assertTrue(Path(default_docs_root()).is_dir())


if __name__ == "__main__":
    unittest.main()
