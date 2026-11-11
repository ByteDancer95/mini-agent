"""具名 Agent 实例 / 协议实录 / 真实 LLM 配置自检。"""
import io
import logging
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

logging.getLogger("mini_agent").setLevel(logging.CRITICAL)

from mini_agent.agent import AGENT_NAME, MiniWorkAgent, create_agent  # noqa: E402
from mini_agent.cli import main as cli_main  # noqa: E402
from mini_agent.core.llm import check_llm_config  # noqa: E402
from mini_agent.walkthrough import build_transcript, main as walkthrough_main  # noqa: E402

EMPTY_ENV = {k: "" for k in [
    "MINI_AGENT_API_KEY", "OPENAI_API_KEY", "MINI_AGENT_BASE_URL", "OPENAI_BASE_URL",
    "MINI_AGENT_MODEL", "OPENAI_MODEL", "MINI_AGENT_LLM_PROTOCOL", "MINI_AGENT_TIMEOUT",
]}
FULL_ENV = {
    "MINI_AGENT_API_KEY": "dummy",
    "MINI_AGENT_BASE_URL": "http://localhost/v1",
    "MINI_AGENT_MODEL": "mock-model",
}


class TestConcreteAgent(unittest.TestCase):
    def test_it_is_a_named_agent_not_a_library(self):
        agent = create_agent(llm_kind="heuristic")
        self.assertIsInstance(agent, MiniWorkAgent)
        self.assertEqual(agent.name, AGENT_NAME)
        for tool in ["calculator", "search", "todo"]:
            self.assertIn(tool, agent.tools())

    def test_describe_marks_whether_decision_is_a_real_llm(self):
        describe = create_agent(llm_kind="heuristic").describe()
        self.assertIn(AGENT_NAME, describe)
        self.assertIn("HeuristicLLM", describe)
        self.assertIn("非真实模型", describe)
        self.assertIn("最大轮次", describe)

    def test_ask_runs_full_loop_with_trace(self):
        agent = create_agent(llm_kind="heuristic")
        result = agent.ask("s1", "算一下 18*24 是多少")
        self.assertEqual(result.status, "ok")
        self.assertIn("432", result.answer)
        self.assertEqual([p["name"] for p in result.trace.tool_calls()], ["calculator"])
        self.assertEqual(result.trace.summary()["tool_calls"], 1)

    def test_multiturn_conversation_keeps_history_and_state(self):
        agent = create_agent(llm_kind="heuristic")
        questions = [
            "查一下广州今天的天气，顺便记个待办：明天给大棚番茄喷药",
            "再帮我算一下 18*24 是多少",
            "继续，把我的待办列出来",
            "不用工具，用一句话说说你刚才都做了什么",
        ]
        turns = [agent.ask("demo", q) for q in questions]
        self.assertEqual(len(turns), 4)
        self.assertTrue(all(r.status == "ok" for r in turns))
        session = agent.runtime.sessions.get_or_create("demo")
        self.assertEqual(len([m for m in session.context.messages if m.role == "user"]), 4)
        self.assertEqual([t["item"] for t in session.state["todos"]],
                         ["明天给大棚番茄喷药"])
        # 最后一轮是纯对话追问，不调工具，且引用了历史
        self.assertEqual(turns[-1].trace.tool_calls(), [])
        self.assertIn("承接上一轮回答", turns[-1].answer)


class TestLLMConfigCheck(unittest.TestCase):
    def test_unconfigured_reports_actionable_error(self):
        with mock.patch.dict(os.environ, EMPTY_ENV):
            status = check_llm_config()
        self.assertFalse(status["ok"])
        self.assertIn("环境变量", status["error"])
        self.assertEqual(set(status.keys()), {"ok", "backend", "error"})
        self.assertNotIn("dummy", str(status))

    def test_configured_reports_backend_without_echoing_secret(self):
        with mock.patch.dict(os.environ, FULL_ENV):
            status = check_llm_config()
        self.assertTrue(status["ok"])
        self.assertEqual(status["backend"], "OpenAICompatLLM")
        self.assertTrue(status["api_key_configured"])
        self.assertNotIn("dummy", str(status))

    def test_cli_check_llm_flag(self):
        with mock.patch.dict(os.environ, EMPTY_ENV), redirect_stdout(io.StringIO()) as buf:
            code = cli_main(["--check-llm"])
        self.assertEqual(code, 1)
        self.assertIn("未配置", buf.getvalue())
        with mock.patch.dict(os.environ, FULL_ENV), redirect_stdout(io.StringIO()) as buf2:
            code2 = cli_main(["--check-llm"])
        self.assertEqual(code2, 0)
        self.assertIn("已配置", buf2.getvalue())


class TestProtocolWalkthrough(unittest.TestCase):
    def test_transcript_contains_full_decision_chain(self):
        text = build_transcript()
        for marker in [
            "工具 Schema 目录", "system prompt", "模型第 1 次原始输出",
            "解析结果", "Runtime 执行工具", "工具结果回灌",
            "第 2 次输出", "最终答案", "AgentRuntime.run()",
            '"name": "calculator"', '"thought"', '"tool_calls"',
            "离线协议验证替身",
        ]:
            self.assertIn(marker, text, marker)

    def test_transcript_can_be_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "transcript.md"
            with redirect_stdout(io.StringIO()):
                code = walkthrough_main(["--save", str(path)])
            self.assertEqual(code, 0)
            self.assertTrue(path.is_file())
            self.assertIn("帧 8", path.read_text(encoding="utf-8"))

    def test_openai_mode_without_config_fails_gracefully(self):
        with mock.patch.dict(os.environ, EMPTY_ENV), redirect_stdout(io.StringIO()) as buf:
            code = walkthrough_main(["--llm", "openai"])
        self.assertEqual(code, 1)
        self.assertIn("未配置", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
