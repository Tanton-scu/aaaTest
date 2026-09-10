"""Agent capability 到处理器的一对一注册表。

Registry 只解决“这个 durable AgentTask 应交给谁”，不做竞价、置信度排序，
也不决定 PriEvO 的 strategy、parent、population 或 budget。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from prievo_agent.domain.models import AgentCapability, AgentTask


class DuplicateAgentRegistrationError(ValueError):
    """同一 capability 或 Agent 名被重复注册。"""


class AgentNotRegisteredError(KeyError):
    """任务要求的 capability 没有对应处理器。"""


@dataclass(frozen=True)
class AgentRegistration:
    """Registry 中的只读注册信息。"""

    name: str
    capability: AgentCapability
    handler: Any

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Agent name 不能为空")


class AgentRegistry:
    """简单、确定性的一对一 capability Registry。

    handler 可以显式包装为 :class:`AgentRegistration`，也可以直接提供
    ``name`` 和 ``capability`` 属性。``resolve`` 返回原始 handler，便于
    Coordinator 直接调用；查询注册元数据时使用 ``registration_for``。
    """

    def __init__(self, agents: Iterable[Any] = ()) -> None:
        self._by_capability: dict[AgentCapability, AgentRegistration] = {}
        self._by_name: dict[str, AgentRegistration] = {}
        for agent in agents:
            self.register(agent)

    def register(
        self,
        agent: Any,
        capability: AgentCapability | str | None = None,
        name: str | None = None,
    ) -> AgentRegistration:
        """注册一个 handler，并拒绝任何模糊或重复映射。"""

        registration = self._registration(agent, capability, name)
        existing_capability = self._by_capability.get(registration.capability)
        if existing_capability is not None:
            raise DuplicateAgentRegistrationError(
                "capability {} 已由 {} 注册".format(
                    registration.capability.value, existing_capability.name
                )
            )
        existing_name = self._by_name.get(registration.name)
        if existing_name is not None:
            raise DuplicateAgentRegistrationError(
                "Agent name {} 已用于 capability {}".format(
                    registration.name, existing_name.capability.value
                )
            )
        self._by_capability[registration.capability] = registration
        self._by_name[registration.name] = registration
        return registration

    def resolve(self, capability: AgentCapability | str) -> Any:
        """返回 capability 唯一对应的 handler。"""

        return self.registration_for(capability).handler

    def registration_for(
        self, capability: AgentCapability | str
    ) -> AgentRegistration:
        normalized = _capability(capability)
        try:
            return self._by_capability[normalized]
        except KeyError as exc:
            raise AgentNotRegisteredError(
                "未注册 capability {}".format(normalized.value)
            ) from exc

    def match(self, task: AgentTask) -> Any:
        """根据 durable AgentTask 的 required_capability 返回 handler。"""

        return self.resolve(task.required_capability)

    def by_name(self, name: str) -> Any:
        try:
            return self._by_name[name].handler
        except KeyError as exc:
            raise AgentNotRegisteredError("未注册 Agent {}".format(name)) from exc

    @property
    def registrations(self) -> tuple[AgentRegistration, ...]:
        """按 capability 值排序，提供稳定、只读的注册快照。"""

        return tuple(
            self._by_capability[item]
            for item in sorted(self._by_capability, key=lambda value: value.value)
        )

    @property
    def capabilities(self) -> frozenset[AgentCapability]:
        return frozenset(self._by_capability)

    def __len__(self) -> int:
        return len(self._by_capability)

    @staticmethod
    def _registration(
        agent: Any,
        capability: AgentCapability | str | None,
        name: str | None,
    ) -> AgentRegistration:
        if isinstance(agent, AgentRegistration):
            if capability is not None or name is not None:
                raise ValueError("AgentRegistration 不允许再覆盖 name/capability")
            return agent

        resolved_capability = capability
        if resolved_capability is None:
            resolved_capability = getattr(agent, "capability", None)
        if resolved_capability is None:
            raise ValueError("Agent 必须声明单一 capability")

        resolved_name = name
        if resolved_name is None:
            resolved_name = getattr(agent, "name", agent.__class__.__name__)
        return AgentRegistration(
            name=str(resolved_name).strip(),
            capability=_capability(resolved_capability),
            handler=agent,
        )


def _capability(value: AgentCapability | str) -> AgentCapability:
    if isinstance(value, AgentCapability):
        return value
    try:
        return AgentCapability(str(value))
    except ValueError as exc:
        raise ValueError("未知 Agent capability：{}".format(value)) from exc
