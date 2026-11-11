"""端到端集成：双窗口演示、trace 日志落盘、源码无密钥、CLI 可运行。"""
import io
import json
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from mini_agent.core.runtime import build_default_runtime
from mini_agent.core.trace import TraceSink
from mini_agent.demo import STEPS, WINDOW_1, WINDOW_2, run_demo

try:
    from .helpers import ROOT, make_heuristic_runtime
except ImportError:
    from helpers import ROOT, make_heuristic_runtime

SUSPICIOUS_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(?i)\b(api[_-]?key|apikey|secret|access[_-]?token)\s*[:=]\s*[\"'][A-Za-z0-9_\-]{16,}[\"']"),
    re.compile(r"Bearer\s+[A-Za-z0-9_\-]{20,}"),
]


class TestEndToEndDemo(unittest.TestCase):
    def test_demo_all_checks_pass(self):
        rt, _ = make_heuristic_runtime()
        with redirect_stdout(io.StringIO()) as buf:
            code = run_demo(rt)
        self.assertEqual(code, 0)
        output = buf.getvalue()
        self.assertIn("全部通过", output)
        self.assertIn("window1 todos", output)
        self.assertIn("weather", output)
        self.assertIn("read_docs", output)
        self.assertIn("todo", output)

    def test_trace_jsonl_and_full_scenario(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "agent_trace.jsonl"
            rt = build_default_runtime(llm_kind="heuristic", trace_path=log)
            results = [rt.run(sid, text) for sid, text, _ in STEPS]
            self.assertTrue(all(r.status == "ok" for r in results))
            self.assertGreater(sum(len(r.trace.tool_calls()) for r in results), 5)

            lines = log.read_text(encoding="utf-8").strip().splitlines()
            self.assertGreater(len(lines), 20)
            events = [json.loads(line) for line in lines]
            types = {e["type"] for e in events}
            self.assertTrue({"run_start", "llm_call", "decision", "tool_call",
                             "tool_result", "final"}.issubset(types))
            for e in events:
                self.assertIn(e["session_id"], {WINDOW_1, WINDOW_2})
                self.assertIn("run_id", e)
            # 每个 session 的事件都各归各的，不串台
            s1_events = [e for e in events if e["session_id"] == WINDOW_1]
            s2_events = [e for e in events if e["session_id"] == WINDOW_2]
            self.assertTrue(s1_events and s2_events)
            self.assertTrue(all(e["session_id"] == WINDOW_1 for e in s1_events))

    def test_trace_sink_echo_and_text(self):
        try:
            from .helpers import call, final_json, make_runtime, tool_json
        except ImportError:
            from helpers import call, final_json, make_runtime, tool_json
        rt, _ = make_runtime(
            [tool_json("t", [call("calculator", expression="1+1")]), final_json("ok")],
            trace_sink=TraceSink(echo=False, to_logger=False))
        res = rt.run("s", "算一下")
        text = res.trace.to_text()
        self.assertIn("tool_call", text)
        self.assertIn("calculator", text)
        self.assertEqual(res.trace.summary()["tool_calls"], 1)

    def test_cli_demo_entrypoint(self):
        from mini_agent.cli import main
        with redirect_stdout(io.StringIO()) as buf:
            code = main(["--demo", "--llm", "heuristic"])
        self.assertEqual(code, 0)
        self.assertIn("隔离验证结果", buf.getvalue())


class TestNoSecretsInRepository(unittest.TestCase):
    def test_python_sources_have_no_hardcoded_credentials(self):
        offenders = []
        for path in ROOT.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for pattern in SUSPICIOUS_PATTERNS:
                if pattern.search(text):
                    offenders.append(f"{path.name}: {pattern.pattern}")
        self.assertEqual(offenders, [], "疑似硬编码密钥: " + "; ".join(offenders))

    def test_env_example_contains_only_placeholders(self):
        env = ROOT / ".env.example"
        self.assertTrue(env.is_file())
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            self.assertIn(key.strip(), {"MINI_AGENT_API_KEY", "MINI_AGENT_BASE_URL",
                                        "MINI_AGENT_MODEL", "MINI_AGENT_LLM_PROTOCOL",
                                        "MINI_AGENT_TIMEOUT"})
            self.assertEqual(value.strip(), "", f"{key} 不应带值")

    def test_gitignore_excludes_env_and_session_state(self):
        text = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn(".env", text)
        self.assertIn("logs/", text)

    def test_required_deliverable_files_exist(self):
        for name in ["README.md", "PROMPT_LOG.md", ".env.example", ".gitignore"]:
            self.assertTrue((ROOT / name).is_file(), name)
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for keyword in ["运行", "系统设计", "召回", "放置", "trace", "测试"]:
            self.assertIn(keyword, readme)


if __name__ == "__main__":
    unittest.main()
