"""多 session 隔离、随时继续对话、持久化恢复。"""
import tempfile
import unittest
from pathlib import Path

from mini_agent.core.session import Session, SessionManager

try:
    from .helpers import make_heuristic_runtime, make_runtime, tool_json, final_json, call
except ImportError:
    from helpers import make_heuristic_runtime, make_runtime, tool_json, final_json, call


class TestSessionIsolation(unittest.TestCase):
    def test_get_or_create_is_idempotent(self):
        mgr = SessionManager()
        a = mgr.get_or_create("A")
        b = mgr.get_or_create("A")
        self.assertIs(a, b)
        self.assertIsNot(a, mgr.get_or_create("B"))
        self.assertEqual(set(mgr.ids()), {"A", "B"})

    def test_empty_session_id_rejected(self):
        with self.assertRaises(ValueError):
            SessionManager().get_or_create("")

    def test_two_windows_are_fully_isolated(self):
        """题目场景：窗口1 查天气记待办 / 窗口2 写周报记待办，互不干扰。"""
        rt, _ = make_heuristic_runtime()
        r1 = rt.run("userA/window1", "查一下广州今天的天气，顺便记个待办：明天给大棚番茄喷药")
        r2 = rt.run("userA/window2", "帮我写一份本周周报，另外记个待办：周五前提交课程报告")
        self.assertTrue(r1.ok and r2.ok)

        s1 = rt.sessions.get_or_create("userA/window1")
        s2 = rt.sessions.get_or_create("userA/window2")
        t1 = [t["item"] for t in s1.state["todos"]]
        t2 = [t["item"] for t in s2.state["todos"]]
        self.assertEqual(t1, ["明天给大棚番茄喷药"])
        self.assertEqual(t2, ["周五前提交课程报告"])

        text1 = "\n".join(m.content for m in s1.context.messages)
        text2 = "\n".join(m.content for m in s2.context.messages)
        self.assertNotIn("周报", text1)
        self.assertNotIn("喷药", text2)
        self.assertNotIn("周五前提交课程报告", text1)
        self.assertNotIn("明天给大棚番茄喷药", text2)

    def test_each_window_can_resume_independently(self):
        rt, _ = make_heuristic_runtime()
        rt.run("w1", "记个待办：A任务")
        rt.run("w2", "记个待办：B任务")
        # 带工具追问：只看得到自己的待办
        r3 = rt.run("w1", "继续，把我刚才记的待办列出来")
        self.assertIn("A任务", r3.answer)
        self.assertNotIn("B任务", r3.answer)
        # 纯对话追问：不调工具，依赖历史上下文
        r4 = rt.run("w2", "继续，不用工具，用一句话复述刚才的内容")
        self.assertNotIn("A任务", r4.answer)
        self.assertEqual(r4.trace.tool_calls(), [])

        s1 = rt.sessions.get_or_create("w1")
        s2 = rt.sessions.get_or_create("w2")
        self.assertEqual(len([m for m in s1.context.messages if m.role == "user"]), 2)
        self.assertEqual(len([m for m in s2.context.messages if m.role == "user"]), 2)
        self.assertEqual(len(s1.state["todos"]), 1)
        self.assertEqual(len(s2.state["todos"]), 1)

    def test_state_objects_are_not_shared(self):
        mgr = SessionManager()
        a, b = mgr.get_or_create("a"), mgr.get_or_create("b")
        a.state["todos"] = [{"id": 1, "item": "only-a"}]
        self.assertEqual(b.state.get("todos"), None)
        self.assertIsNot(a.context, b.context)


class TestPersistence(unittest.TestCase):
    def test_survives_new_session_manager(self):
        with tempfile.TemporaryDirectory() as tmp:
            rt1, _ = make_heuristic_runtime(persist_dir=tmp)
            rt1.run("persisted", "记个待办：跨进程保留的任务")
            rt1.run("persisted", "继续，把我的待办列出来")

            # 模拟进程重启：全新的 SessionManager 指向同一目录
            rt2, _ = make_heuristic_runtime(persist_dir=tmp)
            s = rt2.sessions.get_or_create("persisted")
            users = [m.content for m in s.context.messages if m.role == "user"]
            self.assertEqual(len(users), 2)
            self.assertEqual([t["item"] for t in s.state["todos"]],
                             ["跨进程保留的任务"])
            r = rt2.run("persisted", "继续，把我的待办列出来")
            self.assertIn("跨进程保留的任务", r.answer)

    def test_session_files_are_hashed_no_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = SessionManager(persist_dir=tmp)
            s = mgr.get_or_create("../evil/../../abc")
            mgr.save("../evil/../../abc")
            files = list(Path(tmp).glob("*.json"))
            self.assertEqual(len(files), 1)
            self.assertNotIn("evil", files[0].name)
            self.assertTrue(files[0].name.startswith("session_"))

    def test_delete_and_autosave_toggle(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = SessionManager(persist_dir=tmp)
            mgr.get_or_create("x")
            mgr.save("x")
            self.assertTrue(mgr.exists("x"))
            self.assertTrue(mgr.delete("x"))
            self.assertFalse(mgr.exists("x"))

            mgr2 = SessionManager(persist_dir=tmp, autosave=False)
            mgr2.get_or_create("y")
            mgr2.maybe_save("y")
            self.assertFalse((Path(tmp) / "session_y.json").exists())

    def test_corrupted_persistence_file_is_tolerated(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = SessionManager(persist_dir=tmp)
            mgr.get_or_create("broken")
            mgr.save("broken")
            path = next(Path(tmp).glob("*.json"))
            path.write_text("{ not json", encoding="utf-8")
            fresh = SessionManager(persist_dir=tmp).get_or_create("broken")
            self.assertEqual(fresh.context.messages, [])

    def test_session_serialization_roundtrip(self):
        s = Session("s", None)
        s.state["todos"] = [{"id": 1, "item": "x", "done": False, "priority": "normal"}]
        s.context.add_user("hello")
        s.context.record_thought("thinking", 1)
        restored = Session.from_dict(s.to_dict())
        self.assertEqual(restored.session_id, "s")
        self.assertEqual(restored.state, s.state)
        self.assertEqual(restored.context.messages[0].content, "hello")
        self.assertEqual(restored.context.audit[0]["thought"], "thinking")


if __name__ == "__main__":
    unittest.main()
