"""端到端验证「真实 LLM 链路」：起一个本地 fake OpenAI 兼容端点，
让 AgentRuntime 用 OpenAICompatLLM 完整跑一轮 loop（不需要任何真实密钥）。

这解决笔试题里最容易被质疑的一点：「让 LLM 基于 Schema 自主决策」到底是不是真的跑通过？
这里走的是真实 HTTP 请求 / 响应，不是 mock.patch(urllib.request.urlopen)。

验证链路：
  system prompt 含工具 Schema -> 模型返回 tool_calls -> 解析 -> 执行工具
  -> 工具结果回灌 -> 第二次模型调用 -> 最终答案
"""
from __future__ import annotations

import json
import logging
import os
import threading
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

logging.getLogger("mini_agent").setLevel(logging.CRITICAL)

from mini_agent.core.llm import OpenAICompatLLM  # noqa: E402
from mini_agent.core.runtime import AgentRuntime  # noqa: E402
from mini_agent.core.session import SessionManager  # noqa: E402
from mini_agent.core.trace import TraceSink  # noqa: E402
from mini_agent.tools.base import build_default_registry  # noqa: E402

TOOL_DECISION = {
    "thought": "需要计算，调用 calculator",
    "action": {"type": "tool", "tool_calls": [
        {"name": "calculator", "arguments": {"expression": "18*24"}}]},
}
FINAL_DECISION = {
    "thought": "已拿到计算结果",
    "action": {"type": "final", "content": "计算结果是 432"},
}


@contextmanager
def fake_openai_endpoint(protocol: str = "json"):
    """在本机随机端口起一个最小 OpenAI 兼容服务，收集所有收到的请求体。"""
    received: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            received.append(body)
            has_tool_result = any(m.get("role") == "tool" for m in body.get("messages", []))

            message: dict = {}
            if not has_tool_result and protocol == "native":
                message = {
                    "content": "",
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {
                            "name": "calculator",
                            "arguments": json.dumps({"expression": "18*24"}),
                        },
                    }],
                }
            else:
                decision = FINAL_DECISION if has_tool_result else TOOL_DECISION
                message = {"content": json.dumps(decision, ensure_ascii=False)}

            out = json.dumps({
                "choices": [{"message": message, "finish_reason": "stop"}],
                "model": "mock-model",
            }, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *args):  # 静音 HTTP 访问日志
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        yield base_url, received
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _runtime(llm: OpenAICompatLLM) -> AgentRuntime:
    return AgentRuntime(
        llm=llm,
        registry=build_default_registry(),
        sessions=SessionManager(),
        trace_sink=TraceSink(to_logger=False),
    )


class TestRealLLMProtocolEndToEnd(unittest.TestCase):
    def setUp(self):
        # 避免本机代理把 127.0.0.1 的请求转走
        self.env_patch = mock.patch.dict(os.environ, {"no_proxy": "127.0.0.1,localhost",
                                                      "NO_PROXY": "127.0.0.1,localhost"})
        self.env_patch.start()

    def tearDown(self):
        self.env_patch.stop()

    def test_json_protocol_full_loop_over_http(self):
        with fake_openai_endpoint("json") as (base_url, received):
            llm = OpenAICompatLLM(api_key="test", base_url=base_url, model="mock-model")
            result = _runtime(llm).run("written-test", "帮我算 18*24")

        self.assertEqual(result.status, "ok")
        self.assertIn("432", result.answer)
        self.assertEqual(len(received), 2, "应当发生两次真实 LLM 调用")

        first = received[0]
        system = first["messages"][0]["content"]
        self.assertIn('"name": "calculator"', system, "工具 Schema 必须真的交给模型")
        self.assertIn('"parameters"', system)
        self.assertNotIn("tools", first, "json 协议通过 system prompt 传 Schema")

        second = received[1]
        self.assertTrue(any(m["role"] == "tool" for m in second["messages"]),
                        "工具结果必须回灌进下一轮 context")
        self.assertTrue(any("<tool_result" in m.get("content", "") for m in second["messages"]))
        # 思考过程不应出现在发给模型的任何一轮消息里
        self.assertNotIn("需要计算，调用 calculator", json.dumps(second, ensure_ascii=False))

        self.assertEqual([e["name"] for e in result.trace.tool_calls()], ["calculator"])
        self.assertEqual(len(result.trace.of_type("final")), 1)

    def test_native_function_calling_full_loop_over_http(self):
        with fake_openai_endpoint("native") as (base_url, received):
            llm = OpenAICompatLLM(api_key="test", base_url=base_url,
                                  model="mock-model", protocol="native")
            result = _runtime(llm).run("written-test", "帮我算 18*24")

        self.assertEqual(result.status, "ok")
        self.assertIn("432", result.answer)
        self.assertEqual(len(received), 2)
        tools = received[0].get("tools")
        self.assertTrue(tools, "native 协议必须通过 tools 参数传 Schema")
        self.assertEqual(tools[0]["function"]["name"], "calculator")
        self.assertEqual([e["name"] for e in result.trace.tool_calls()], ["calculator"])
        self.assertEqual(result.trace.of_type("parse")[0].payload["source"], "native")

    def test_backend_is_reported_as_real_llm(self):
        from mini_agent.agent import MiniWorkAgent
        with fake_openai_endpoint("json") as (base_url, _):
            llm = OpenAICompatLLM(api_key="test", base_url=base_url, model="mock-model")
            agent = MiniWorkAgent(llm=llm)
            self.assertTrue(agent.is_real_llm)
            self.assertIn("真实 LLM", agent.describe())


if __name__ == "__main__":
    unittest.main()
