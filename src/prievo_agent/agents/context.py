from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BuiltAgentContext:
    text: str
    max_chars: int
    chars_before: int
    chars_after: int
    compression_applied: bool
    omitted_items: int

    def metadata(self):
        values = asdict(self)
        values.pop("text")
        return values


class AgentContextBuilder:
    """按优先级构造有界 Prompt；pinned 事实永不被静默丢弃。"""

    def __init__(self, max_chars=6000):
        self.max_chars = max(800, int(max_chars))

    def build(self, pinned, recent_memory=None, long_term_memory=None, evidence=None):
        sections = [
            ("Pinned Agent Context", [str(item) for item in pinned], True),
            ("Literature evidence", [str(item) for item in (evidence or [])], False),
            ("Recent Agent Working Memory", [str(item) for item in (recent_memory or [])], False),
            ("Relevant Long-Term Agent Memory", [str(item) for item in (long_term_memory or [])], False),
        ]
        full_text = self._render([(name, items) for name, items, _ in sections if items])
        if len(full_text) <= self.max_chars:
            return BuiltAgentContext(
                full_text + "\n[Context compression: applied=false]",
                self.max_chars, len(full_text), len(full_text), False, 0,
            )

        kept = []
        omitted = 0
        for name, items, pinned_section in sections:
            if not items:
                continue
            if pinned_section:
                kept.append((name, items))
                continue
            accepted = []
            for item in items:
                candidate = self._render([*kept, (name, [*accepted, item])])
                reserve = 120
                if len(candidate) <= self.max_chars - reserve:
                    accepted.append(item)
                else:
                    omitted += 1
            if accepted:
                kept.append((name, accepted))
        text = self._render(kept)
        footer = (
            "\n[Context compression: applied=true; omitted_items={}; "
            "chars_before={}; max_chars={}]"
        ).format(omitted, len(full_text), self.max_chars)
        # pinned 内容可能自身超预算；此时显式保留并暴露超限，绝不悄悄截断事实。
        text += footer
        return BuiltAgentContext(
            text, self.max_chars, len(full_text), len(text), True, omitted,
        )

    @staticmethod
    def _render(sections):
        return "\n".join(
            "{}:\n{}".format(name, "\n".join("- " + item for item in items))
            for name, items in sections if items
        )

