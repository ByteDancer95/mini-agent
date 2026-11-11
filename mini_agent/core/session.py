"""多 Session 隔离 + 可选持久化。

隔离保证
--------
* 每个 Session 独享 ``ContextManager``（历史）与 ``state``（todo 等有状态工具的数据）。
* Runtime 每次执行工具都构造 ``ToolContext(session_id, state=session.state, ...)``，
  工具只碰传进来的 state，不碰任何模块级全局变量。
* ``SessionManager`` 只做 id -> Session 的映射，跨 session 没有任何共享可变对象。

随时继续对话
------------
* 进程内：切回同一个 session_id 立即拿到原历史；
* 跨进程：开启 ``persist_dir`` 后，每次 run 结束落盘 JSON，下次启动自动加载。
文件名用 session_id 的 sha256，避免任意字符串直接当路径造成穿越。
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .context import ContextManager, ContextPolicy

__all__ = ["Session", "SessionManager"]


class Session:
    def __init__(self, session_id: str, policy: ContextPolicy | None = None) -> None:
        self.session_id = session_id
        self.context = ContextManager(policy)
        self.state: dict[str, Any] = {}
        self.created_at = time.time()
        self.updated_at = time.time()
        self.meta: dict[str, Any] = {}

    def touch(self) -> None:
        self.updated_at = time.time()

    def state_snapshot(self) -> str:
        """会话状态的紧凑快照（按需注入 context，不是每轮全量回放）。"""
        todos = self.state.get("todos") or []
        if not todos:
            return ""
        lines = [f"todos ({len(todos)}):"]
        for t in todos:
            mark = "x" if t.get("done") else " "
            lines.append(f"  [{mark}] #{t.get('id')} {t.get('item')} (priority={t.get('priority', 'normal')})")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "state": self.state,
            "context": self.context.to_dict(),
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], policy: ContextPolicy | None = None) -> "Session":
        s = cls(str(data["session_id"]), policy)
        s.created_at = float(data.get("created_at", time.time()))
        s.updated_at = float(data.get("updated_at", time.time()))
        s.state = dict(data.get("state") or {})
        s.context = ContextManager.from_dict(data.get("context") or {})
        if policy is not None:
            s.context.policy = policy
        s.meta = dict(data.get("meta") or {})
        return s


class SessionManager:
    def __init__(
        self,
        persist_dir: str | Path | None = None,
        policy: ContextPolicy | None = None,
        autosave: bool = True,
    ) -> None:
        self.persist_dir = Path(persist_dir) if persist_dir else None
        if self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.policy = policy
        self.autosave = autosave
        self._sessions: dict[str, Session] = {}

    # ---- 路径 -------------------------------------------------------------
    def _path_for(self, session_id: str) -> Path | None:
        if not self.persist_dir:
            return None
        digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:24]
        return self.persist_dir / f"session_{digest}.json"

    # ---- 基本操作 ---------------------------------------------------------
    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def get_or_create(self, session_id: str) -> Session:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id 必须是非空字符串")
        if session_id in self._sessions:
            return self._sessions[session_id]

        path = self._path_for(session_id)
        if path and path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("session_id") == session_id:
                    session = Session.from_dict(data, self.policy)
                    self._sessions[session_id] = session
                    return session
            except (ValueError, OSError, KeyError, TypeError):
                # 损坏的持久化文件不应让整个 Agent 起不来
                pass
        session = Session(session_id, self.policy)
        self._sessions[session_id] = session
        return session

    def ids(self) -> list[str]:
        return list(self._sessions.keys())

    def exists(self, session_id: str) -> bool:
        if session_id in self._sessions:
            return True
        path = self._path_for(session_id)
        return bool(path and path.is_file())

    def delete(self, session_id: str) -> bool:
        existed = self._sessions.pop(session_id, None) is not None
        path = self._path_for(session_id)
        removed_file = False
        if path and path.is_file():
            try:
                path.unlink()
                removed_file = True
            except OSError:
                pass
        return existed or removed_file

    # ---- 持久化 -----------------------------------------------------------
    def save(self, session_id: str) -> bool:
        session = self._sessions.get(session_id)
        path = self._path_for(session_id)
        if session is None or path is None:
            return False
        try:
            payload = json.dumps(session.to_dict(), ensure_ascii=False, indent=2, default=str)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(path)
            return True
        except (OSError, TypeError, ValueError):
            return False

    def save_all(self) -> int:
        return sum(1 for sid in self.ids() if self.save(sid))

    def maybe_save(self, session_id: str) -> None:
        """autosave 开启时落盘；失败不影响主流程。"""
        if self.autosave:
            self.save(session_id)
