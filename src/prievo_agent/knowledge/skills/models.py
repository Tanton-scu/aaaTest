from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SkillDefinition:
    name: str
    version: str
    purpose: str
    instructions: str
    content_digest: str


class SkillRegistryPort(Protocol):
    def require(self, name: str) -> SkillDefinition:
        ...
