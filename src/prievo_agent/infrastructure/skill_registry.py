from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict

from prievo_agent.domain.skills import SkillDefinition


class SkillLoadError(RuntimeError):
    pass


class SkillRegistry:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._skills = self._load()

    def require(self, name: str) -> SkillDefinition:
        try:
            return self._skills[name]
        except KeyError as exc:
            raise SkillLoadError("Skill 不存在：{}".format(name)) from exc

    def _load(self) -> Dict[str, SkillDefinition]:
        result = {}
        for path in sorted(self.root.glob("*/SKILL.md")):
            text = path.read_text(encoding="utf-8")
            metadata, instructions = self._split(text, path)
            name = metadata.get("name", "").strip()
            if not name or name in result:
                raise SkillLoadError("Skill name 缺失或重复：{}".format(path))
            result[name] = SkillDefinition(
                name,
                metadata.get("version", "1"),
                metadata.get("purpose", ""),
                instructions.strip(),
                hashlib.sha256(text.encode("utf-8")).hexdigest(),
            )
        return result

    @staticmethod
    def _split(text: str, path: Path):
        if not text.startswith("---\n"):
            raise SkillLoadError("Skill 缺少 frontmatter：{}".format(path))
        _, frontmatter, body = text.split("---", 2)
        metadata = {}
        for line in frontmatter.strip().splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()
        return metadata, body
