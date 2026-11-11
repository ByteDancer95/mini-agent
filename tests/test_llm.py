"""LLM 客户端：环境变量配置、无密钥降级、OpenAI 兼容协议（含 native tools）、输出解析链路。"""
import io
import json
import logging
import os
import tempfile
import unittest
import urllib.error
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

logging.getLogger("mini_agent").setLevel(logging.CRITICAL)

from mini_agent.core.errors import LLMConfigError, LLMError
from mini_agent.core.llm import (HeuristicLLM, OpenAICompatLLM, ScriptedLLM,
                                 build_llm, load_env_file)
from mini_agent.core.types import Message, Role

_ENV_KEYS = ["MINI_AGENT_API_KEY", "OPENAI_API_KEY", "MINI_AGENT_BASE_URL",
             "OPENAI_BASE_URL", "MINI_AGENT_MODEL", "OPENAI_MODEL",
             "MINI_AGENT_LLM_PROTOCOL", "MINI_AGENT_TIMEOUT"]


def _empty_env():
    return mock.patch.dict(os.environ, {k: "" for k in _ENV_KEYS})


class FakeHTTPResponse:
    def __init__(self, payload: dict):
        self._raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _chat_response(content="", tool_calls=None):
    message = {"content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message, "finish_reason": "stop"}], "model": "mock-model"}


class TestConfigSafety(unittest.TestCase):
    def test_missing_config_raises_actionable_error(self):
        with _empty_env():
            with self.assertRaises(LLMConfigError) as ctx:
                OpenAICompatLLM()
        self.assertIn("环境变量", str(ctx.exception))
        self.assertIn("MINI_AGENT_API_KEY", str(ctx.exception))

    def test_no_literal_credentials_in_source(self):
        """保证 LLM 端点/密钥只来自环境变量，不硬编码在代码里。"""
        import mini_agent.core.llm as mod
        source = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertNotIn("api.openai.com", source)
        self.assertNotIn("sk-", source)
        self.assertIn("os.environ", source)

    def test_auto_falls_back_to_zero_key_demo_mode(self):
        with _empty_env(), redirect_stdout(io.StringIO()):
            llm = build_llm("auto")
        self.assertIsInstance(llm, HeuristicLLM)
        self.assertIsInstance(build_llm("heuristic"), HeuristicLLM)

    def test_load_env_file_never_overrides_process_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text(
                "# comment\nMINI_AGENT_MODEL=from_file\nA_NEW_TEST_VAR=loaded\n",
                encoding="utf-8")
            with mock.patch.dict(os.environ, {"MINI_AGENT_MODEL": "from_env"}, clear=False):
                count = load_env_file(env_file)
                self.assertEqual(os.environ["MINI_AGENT_MODEL"], "from_env")
                self.assertEqual(os.environ.get("A_NEW_TEST_VAR"), "loaded")
                self.assertGreaterEqual(count, 1)
            os.environ.pop("A_NEW_TEST_VAR", None)

    def test_env_file_missing_is_noop(self):
        self.assertEqual(load_env_file("/definitely/not/here/.env"), 0)


class TestOpenAICompatClient(unittest.TestCase):
    CONFIG = {
        "MINI_AGENT_API_KEY": "dummy",
        "MINI_AGENT_BASE_URL": "http://localhost/v1",
        "MINI_AGENT_MODEL": "mock-model",
    }
    SCHEMAS = [{"name": "todo", "description": "待办",
                "parameters": {"type": "object", "properties": {}}}]

    def _client(self, **kwargs):
        with mock.patch.dict(os.environ, self.CONFIG):
            return OpenAICompatLLM(**kwargs)

    def test_json_protocol_request_payload(self):
        client = self._client()
        fake = FakeHTTPResponse(_chat_response('{"action": {"type": "final", "content": "hi"}}'))
        with mock.patch("mini_agent.core.llm.urllib.request.urlopen", return_value=fake) as u:
            resp = client.generate([Message("system", "SYS"), Message("user", "hello")],
                                   self.SCHEMAS, False)
        self.assertEqual(resp.raw, '{"action": {"type": "final", "content": "hi"}}')
        req = u.call_args[0][0]
        self.assertTrue(req.full_url.endswith("/chat/completions"))
        payload = json.loads(req.data.decode("utf-8"))
        self.assertEqual(payload["model"], "mock-model")
        self.assertEqual([m["role"] for m in payload["messages"]], ["system", "user"])
        self.assertNotIn("tools", payload)  # json 协议把目录放进 system prompt
        self.assertIn("Authorization", req.headers)
        # 请求体里不应出现密钥
        self.assertNotIn("dummy", json.dumps(payload))

    def test_native_protocol_sends_tools_and_parses_tool_calls(self):
        client = self._client(protocol="native")
        tc = [{"id": "c1", "function": {"name": "todo",
                                        "arguments": '{"action": "list"}'}}]
        fake = FakeHTTPResponse(_chat_response("", tool_calls=tc))
        with mock.patch("mini_agent.core.llm.urllib.request.urlopen", return_value=fake) as u:
            resp = client.generate([Message("user", "列出待办")], self.SCHEMAS, False)
        self.assertEqual(client.protocol_name, "native")
        payload = json.loads(u.call_args[0][0].data.decode("utf-8"))
        self.assertEqual(payload["tools"][0]["function"]["name"], "todo")
        self.assertEqual(len(resp.native_tool_calls), 1)
        self.assertEqual(resp.native_tool_calls[0].arguments, {"action": "list"})

    def test_force_final_disables_native_tools(self):
        client = self._client(protocol="native")
        fake = FakeHTTPResponse(_chat_response("done"))
        with mock.patch("mini_agent.core.llm.urllib.request.urlopen", return_value=fake) as u:
            client.generate([Message("user", "x")], self.SCHEMAS, force_final=True)
        payload = json.loads(u.call_args[0][0].data.decode("utf-8"))
        self.assertEqual(payload["tool_choice"], "none")

    def test_http_error_becomes_llm_error_without_leaking_key(self):
        client = self._client()
        err = urllib.error.HTTPError("http://localhost/v1/chat/completions", 401,
                                     "Unauthorized", {}, io.BytesIO(b"bad key"))
        with mock.patch("mini_agent.core.llm.urllib.request.urlopen", side_effect=err):
            with self.assertRaises(LLMError) as ctx:
                client.generate([Message("user", "x")], self.SCHEMAS, False)
        self.assertIn("401", str(ctx.exception))
        self.assertNotIn("dummy", str(ctx.exception))

    def test_network_error_becomes_llm_error(self):
        client = self._client()
        with mock.patch("mini_agent.core.llm.urllib.request.urlopen",
                        side_effect=urllib.error.URLError("offline")):
            with self.assertRaises(LLMError):
                client.generate([Message("user", "x")], self.SCHEMAS, False)


class TestLocalLLMs(unittest.TestCase):
    def test_heuristic_decides_multiple_tools_then_finalizes(self):
        llm = HeuristicLLM()
        first = llm.generate([Message("user", "查一下广州天气，记个待办：喷药")], [], False)
        self.assertIn("weather", first.raw)
        self.assertIn("todo", first.raw)
        second = llm.generate([
            Message("user", "查一下广州天气"),
            Message("assistant", "decision"),
            Message("tool", '<tool_result name="weather">{"ok":true,"result":'
                            '{"city":"广州","condition":"晴","temperature":26}}</tool_result>'),
        ], [], False)
        self.assertIn("final", second.raw)
        self.assertEqual(len(llm.calls), 2)

    def test_heuristic_pure_conversation_needs_no_tool(self):
        llm = HeuristicLLM()
        resp = llm.generate([Message("user", "不用工具，随便聊聊")], [], False)
        self.assertIn("final", resp.raw)
        self.assertNotIn("tool_calls", resp.raw)

    def test_scripted_llm_records_messages_and_exhaustion(self):
        llm = ScriptedLLM(['{"final_answer":"A"}'], default_final="B")
        r1 = llm.generate([Message("user", "1")], [], False)
        r2 = llm.generate([Message("user", "2")], [], True)
        self.assertIn("A", r1.raw)
        self.assertIn("B", r2.raw)
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(llm.force_final_flags, [False, True])


if __name__ == "__main__":
    unittest.main()
