"""工具注册机制 / Schema 校验 / 安全执行边界。"""
import unittest

from mini_agent.core.registry import ToolRegistry, validate_against_schema
from mini_agent.core.types import ToolCall, ToolContext, ToolSpec
from mini_agent.tools.base import build_default_registry, build_minimal_registry

try:
    from .helpers import ROOT
except ImportError:  # python -m unittest discover -s tests 会把测试模块当顶层模块加载
    from helpers import ROOT


class TestToolRegistration(unittest.TestCase):
    def test_three_required_tools_exist(self):
        reg = build_minimal_registry()
        self.assertEqual(set(reg.names()), {"calculator", "search", "todo"})
        self.assertEqual(len(reg), 3)

    def test_default_registry_includes_extensions(self):
        reg = build_default_registry()
        self.assertEqual(set(reg.names()),
                         {"calculator", "search", "todo", "weather", "read_docs"})

    def test_extensions_are_optional(self):
        reg = build_default_registry(extensions=["weather"])
        self.assertIn("weather", reg)
        self.assertNotIn("read_docs", reg)
        with self.assertRaises(ValueError):
            build_default_registry(extensions=["not_exist"])

    def test_every_spec_has_name_description_and_schema(self):
        for schema in build_default_registry().schemas():
            self.assertEqual(set(schema.keys()), {"name", "description", "parameters"})
            self.assertIsInstance(schema["name"], str)
            self.assertTrue(schema["description"].strip())
            self.assertEqual(schema["parameters"]["type"], "object")
            self.assertIn("properties", schema["parameters"])

    def test_native_schema_shape(self):
        native = build_default_registry().native_schemas()[0]
        self.assertEqual(native["type"], "function")
        self.assertIn("parameters", native["function"])

    def test_decorator_custom_extension(self):
        """题目要求的「允许自定义扩展」：注册新工具后 Runtime 无需改动。"""
        reg = build_minimal_registry()

        @reg.tool("echo", "回显文本", {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        })
        def echo(args, ctx):
            return {"text": args["text"], "session": ctx.session_id}

        self.assertIn("echo", reg.names())
        result = reg.execute(ToolCall("echo", {"text": "hi"}), ToolContext("s1"))
        self.assertTrue(result.ok)
        self.assertEqual(result.data["text"], "hi")
        self.assertEqual(result.data["session"], "s1")
        # 新工具的 Schema 自动出现在给 LLM 的目录里
        self.assertIn("echo", [s["name"] for s in reg.schemas()])

    def test_duplicate_registration_rejected(self):
        reg = build_minimal_registry()
        spec = ToolSpec("dup", "d", {"type": "object", "properties": {}})
        reg.register(spec, lambda a, c: {})
        with self.assertRaises(ValueError):
            reg.register(spec, lambda a, c: {})
        reg.register(spec, lambda a, c: {}, replace=True)
        reg.unregister("dup")
        self.assertNotIn("dup", reg)

    def test_bad_registration_inputs(self):
        reg = ToolRegistry()
        with self.assertRaises(TypeError):
            reg.register("not a spec", lambda a, c: {})  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            reg.register(ToolSpec("x", "d", {"type": "object"}), "not callable")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            reg.register(ToolSpec("", "d", {"type": "object"}), lambda a, c: {})


class TestSchemaValidation(unittest.TestCase):
    def setUp(self):
        self.reg = build_default_registry()

    def test_unknown_tool_returns_error_result_not_exception(self):
        res = self.reg.execute(ToolCall("does_not_exist", {}), ToolContext("s"))
        self.assertFalse(res.ok)
        self.assertIn("unknown_tool", res.error)
        self.assertIn("calculator", res.error)

    def test_missing_required_argument(self):
        res = self.reg.execute(ToolCall("calculator", {}), ToolContext("s"))
        self.assertFalse(res.ok)
        self.assertIn("参数校验失败", res.error)
        self.assertIn("expression", res.error)

    def test_wrong_type(self):
        res = self.reg.execute(ToolCall("calculator", {"expression": 123}), ToolContext("s"))
        self.assertFalse(res.ok)
        self.assertIn("类型错误", res.error)

    def test_enum_constraint(self):
        res = self.reg.execute(ToolCall("todo", {"action": "explode"}), ToolContext("s"))
        self.assertFalse(res.ok)
        self.assertIn("不在允许集合", res.error)

    def test_additional_properties_false(self):
        res = self.reg.execute(
            ToolCall("calculator", {"expression": "1+1", "malicious": True}), ToolContext("s"))
        self.assertFalse(res.ok)
        self.assertIn("未声明参数", res.error)

    def test_bounds(self):
        res = self.reg.execute(
            ToolCall("search", {"query": "x", "top_k": 99}), ToolContext("s"))
        self.assertFalse(res.ok)
        self.assertIn("maximum", res.error)

    def test_handler_exception_is_isolated(self):
        reg = ToolRegistry()

        def boom(args, ctx):
            raise RuntimeError("boom!")

        reg.register(ToolSpec("boom", "always fails", {"type": "object", "properties": {}}), boom)
        res = reg.execute(ToolCall("boom", {}), ToolContext("s"))
        self.assertFalse(res.ok)
        self.assertIn("工具执行异常", res.error)
        self.assertIn("boom!", res.error)
        self.assertGreaterEqual(res.elapsed_ms, 0)

    def test_handler_returning_error_dict(self):
        reg = ToolRegistry()
        reg.register(ToolSpec("bad", "x", {"type": "object", "properties": {}}),
                     lambda a, c: {"ok": False, "error": "custom failure"})
        res = reg.execute(ToolCall("bad", {}), ToolContext("s"))
        self.assertFalse(res.ok)
        self.assertEqual(res.error, "custom failure")

    def test_standalone_validator(self):
        schema = {"type": "object", "properties": {"n": {"type": "integer", "minimum": 1}},
                  "required": ["n"]}
        validate_against_schema(schema, {"n": 2})
        with self.assertRaises(Exception):
            validate_against_schema(schema, {"n": True})
        with self.assertRaises(Exception):
            validate_against_schema(schema, {"n": 0})


if __name__ == "__main__":
    unittest.main()
