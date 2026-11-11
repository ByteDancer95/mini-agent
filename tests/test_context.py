"""Context 管理：召回时机、放置方式、思考过程去向、基础压缩。"""
import logging
import unittest

logging.getLogger("mini_agent").setLevel(logging.CRITICAL)

from mini_agent.core.context import ContextManager, ContextPolicy, tool_result_text
from mini_agent.core.types import AgentOutput, Role, ToolCall, ToolResult
from mini_agent.core.parser import parse_llm_output


class TestContextWritingPolicy(unittest.TestCase):
    def test_thought_not_written_into_messages_by_default(self):
        ctx = ContextManager()
        out = AgentOutput(thought="PRIVATE_THOUGHT",
                          tool_calls=[ToolCall("calculator", {"expression": "1+1"})])
        msg = ctx.add_decision(out)
        self.assertNotIn("PRIVATE_THOUGHT", msg.content)
        self.assertIn("calculator", msg.content)
        ctx.record_thought(out.thought, 1)
        self.assertEqual(ctx.audit[0]["thought"], "PRIVATE_THOUGHT")
        self.assertNotIn("PRIVATE_THOUGHT", "\n".join(m.content for m in ctx.messages))

    def test_thought_included_when_policy_allows(self):
        ctx = ContextManager(ContextPolicy(include_thought_in_llm_context=True))
        msg = ctx.add_decision(AgentOutput(
            thought="VISIBLE_THOUGHT",
            tool_calls=[ToolCall("todo", {"action": "list"})]))
        self.assertIn("VISIBLE_THOUGHT", msg.content)

    def test_tool_result_placement_and_meta(self):
        ctx = ContextManager()
        msg = ctx.add_tool_result(ToolCall("calculator", {"expression": "1+1"}),
                                  ToolResult(ok=True, data={"result": 2}))
        self.assertEqual(msg.role, Role.TOOL.value)
        self.assertIn("<tool_result", msg.content)
        self.assertEqual(msg.meta["tool"], "calculator")
        self.assertEqual(msg.meta["payload"]["result"]["result"], 2)

    def test_tool_result_truncation(self):
        ctx = ContextManager(ContextPolicy(tool_result_max_chars=60))
        msg = ctx.add_tool_result(ToolCall("search", {}),
                                  ToolResult(ok=True, data={"blob": "x" * 500}))
        self.assertIn("truncated", msg.content)
        self.assertLess(len(msg.content), 200)

    def test_message_order(self):
        ctx = ContextManager()
        ctx.add_user("问题")
        ctx.add_decision(AgentOutput(tool_calls=[ToolCall("todo", {"action": "list"})]))
        ctx.add_tool_result(ToolCall("todo", {"action": "list"}),
                            ToolResult(ok=True, data={"todos": []}))
        ctx.add_final("答案")
        self.assertEqual([m.role for m in ctx.messages],
                         ["user", "assistant", "tool", "assistant"])


class TestContextRecallPlacement(unittest.TestCase):
    def test_build_messages_order_system_digest_snapshot_history(self):
        ctx = ContextManager(ContextPolicy(max_messages=2, keep_recent=1))
        for i in range(4):
            ctx.add_user(f"u{i}")
        self.assertTrue(ctx.maybe_compact())
        messages = ctx.build_messages("SYS_PROMPT", "TODOS_SNAPSHOT")
        kinds = [m.meta.get("kind") for m in messages]
        self.assertEqual(kinds[:3], ["system_prompt", "digest", "state_snapshot"])
        self.assertEqual(messages[0].content, "SYS_PROMPT")
        self.assertIn("TODOS_SNAPSHOT", messages[2].content)
        self.assertTrue(all(m.role == "system" for m in messages[:3]))
        # 工具目录/提示词每次重建，不落在历史里
        self.assertNotIn("system_prompt", [m.meta.get("kind") for m in ctx.messages])

    def test_snapshot_not_injected_when_disabled_or_empty(self):
        ctx = ContextManager(ContextPolicy(inject_state_snapshot=False))
        ctx.add_user("hi")
        self.assertFalse(any("TODOS" in m.content for m in ctx.build_messages("S", "TODOS")))
        ctx2 = ContextManager()
        ctx2.add_user("hi")
        self.assertFalse(any(m.meta.get("kind") == "state_snapshot"
                             for m in ctx2.build_messages("S", None)))

    def test_digest_is_recalled_as_system_message(self):
        ctx = ContextManager(ContextPolicy(max_messages=2, keep_recent=1))
        ctx.add_user("EARLY_QUESTION")
        ctx.add_final("EARLY_ANSWER")
        ctx.add_user("LATE_QUESTION")
        ctx.maybe_compact()
        messages = ctx.build_messages("S")
        self.assertTrue(any("Earlier conversation digest" in m.content for m in messages))
        self.assertTrue(any("EARLY_QUESTION" in m.content for m in messages))


class TestBasicCompression(unittest.TestCase):
    def test_compaction_by_message_count(self):
        ctx = ContextManager(ContextPolicy(max_messages=5, max_chars=10 ** 6, keep_recent=2))
        for i in range(8):
            ctx.add_user(f"msg-{i}")
        self.assertTrue(ctx.needs_compaction())
        self.assertTrue(ctx.maybe_compact())
        self.assertEqual(len(ctx.messages), 2)
        self.assertEqual(len(ctx.digest), 6)
        self.assertEqual(ctx.stats()["compactions"], 1)
        self.assertFalse(ctx.needs_compaction())

    def test_compaction_by_char_budget(self):
        ctx = ContextManager(ContextPolicy(max_messages=100, max_chars=80, keep_recent=2))
        for i in range(4):
            ctx.add_user("x" * 50)
        self.assertTrue(ctx.needs_compaction())
        self.assertTrue(ctx.maybe_compact())

    def test_compression_can_be_disabled(self):
        ctx = ContextManager(ContextPolicy(compress=False, max_messages=1))
        for i in range(5):
            ctx.add_user(str(i))
        self.assertFalse(ctx.needs_compaction())
        self.assertFalse(ctx.maybe_compact())
        self.assertEqual(len(ctx.messages), 5)

    def test_compaction_keeps_most_recent_user_message(self):
        ctx = ContextManager(ContextPolicy(max_messages=3, keep_recent=2))
        ctx.add_user("ONLY_USER")
        ctx.add_final("a")
        ctx.add_tool_result(ToolCall("todo", {"action": "list"}),
                            ToolResult(ok=True, data={"todos": []}))
        ctx.add_tool_result(ToolCall("weather", {"city": "广州"}),
                            ToolResult(ok=True, data={"temperature": 26}))
        ctx.add_final("b")
        ctx.maybe_compact()
        self.assertTrue(any(m.role == "user" and m.content == "ONLY_USER"
                            for m in ctx.messages))

    def test_digest_is_bounded(self):
        ctx = ContextManager(ContextPolicy(max_messages=2, keep_recent=1,
                                           digest_max_lines=3))
        for i in range(10):
            ctx.add_user(f"m{i}")
        ctx.maybe_compact()
        # 一次压缩后 digest 最多保留 digest_max_lines 行
        self.assertLessEqual(len(ctx.digest), 3)

    def test_stats_and_serialization_roundtrip(self):
        ctx = ContextManager(ContextPolicy(max_messages=3, keep_recent=1))
        ctx.add_user("q")
        ctx.record_thought("thinking", 1, "json")
        ctx.add_final("a")
        ctx.add_user("q2")
        ctx.maybe_compact()
        restored = ContextManager.from_dict(ctx.to_dict())
        self.assertEqual(restored.stats(), ctx.stats())
        self.assertEqual([m.content for m in restored.messages],
                         [m.content for m in ctx.messages])
        self.assertEqual(restored.audit, ctx.audit)
        self.assertEqual(restored.digest, ctx.digest)
        self.assertEqual(restored.policy.max_messages, 3)

    def test_tool_result_text_helper(self):
        text = tool_result_text("x", "cid", {"ok": True, "result": 1}, limit=200)
        self.assertIn('name="x"', text)
        self.assertIn('"ok": true', text)


if __name__ == "__main__":
    unittest.main()
