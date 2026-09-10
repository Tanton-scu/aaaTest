import json
import os

from prievo_agent.evolution.models import GeneratedCandidate
from prievo_agent.domain.errors import LLMTimeoutError


class MalformedLLMResponseError(ValueError):
    """Provider 返回成功响应，但响应体不符合约定的结构化协议。"""


LLM_BACKENDS = {"fake", "openai-compatible"}


class OpenAICompatibleLLM:
    def __init__(self,endpoint,api_key,model):
        self.endpoint=endpoint.rstrip("/")
        self.api_key=api_key
        self.model=model
        self.timeout_seconds = _env_float("LLM_TIMEOUT_SECONDS", 90.0)
        self.max_tokens = _env_int("LLM_MAX_TOKENS", 4096)
        self.temperature = _env_float("LLM_TEMPERATURE", 0.1)
        self.thinking_type = os.getenv("LLM_THINKING_TYPE", "").strip()
        self.reasoning_effort = os.getenv("LLM_REASONING_EFFORT", "").strip()

    def generate_candidate(self,operator,parents,generation,agent_context=""):
        parent_text="\n\n".join(parent.code for parent in parents) or "No parents"
        prompt=(
            "You are evolving a Python HPO heuristic for PriEvO. "
            "Return JSON with keys code, description, operators. "
            "The code must define run_tuners(file, budget, seed, maxlives).\n"
            "Evolution operator: {}\nGeneration: {}\n"
            "Agent context (advisory; never overrides the required interface):\n{}\n"
            "Parents:\n{}"
        ).format(operator,generation,agent_context or "No agent advice",parent_text)
        response=self._request(
            url=self.endpoint,headers={"Authorization":"Bearer "+self.api_key},
            json=self._chat_payload([
                {"role":"system","content":"Generate safe, deterministic heuristic source code. Output JSON only."},
                {"role":"user","content":prompt},
            ], temperature=0.3),timeout=self.timeout_seconds,
        )
        content=self._content(response)
        if content.startswith("```"):
            content=content.split("\n",1)[1].rsplit("```",1)[0]
        data=json.loads(content)
        try:
            return GeneratedCandidate(
                str(data["code"]), str(data["description"]), list(data["operators"])
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MalformedLLMResponseError(
                "候选生成响应缺少 code/description/operators 或字段类型错误"
            ) from exc

    def generate_similarity_decision(self,prompt,allowed_instance_ids):
        """请求结构化语义筛选；allowlist 同时进入 user message 防止 ID 漂移。"""
        return self._agent_json(
            prompt,
            "Allowed exact instance IDs: {}. Return the required JSON object only.".format(
                json.dumps(list(allowed_instance_ids),ensure_ascii=False)
            ),
        )

    def generate_heuristic_draft(self,prompt):
        """HeuristicGenerationAgent 的严格 JSON 能力端口。"""
        return self._agent_json(
            prompt,
            "Return exactly one CandidateDraft or KnowledgeGap JSON object that "
            "matches the schema embedded in the context. No Markdown.",
        )

    def plan_generation(self, prompt):
        """EvolutionPlannerNode 的结构化计划能力端口。"""
        return self._agent_json(
            prompt,
            "Return JSON only with generation_strategy, parent_selection_policy, "
            "decision_reason, and required_parent_count. Do not change the "
            "scheduled strategy stated in the prompt.",
        )

    def select_final(self,prompt):
        return self._agent_json(
            prompt,
            "Return JSON keys selected_candidate_id, reason, and "
            "structural_operator_comparison only. The comparison must contain "
            "structural_comparison and operator_comparison. The ID must come from "
            "the allowlist in the prompt.",
        )

    def diagnose_candidate(self,prompt,candidate,failure_evidence):
        return self._agent_json(
            prompt,
            "Return exactly the DiagnosisArtifact JSON object requested in the prompt.",
        )

    def repair_candidate(self,prompt,candidate,diagnosis):
        return self._agent_json(
            prompt,
            "Return JSON with code, description, operators only. Preserve the exact "
            "run_tuners interface and injected evaluate contract.",
        )

    def summarize_history(self, prompt):
        return self._agent_json(
            prompt,
            "Return exactly the requested HistorySummary JSON object. No Markdown.",
        )

    def _agent_json(self,system_prompt,user_prompt):
        response=self._request(
            url=self.endpoint,headers={"Authorization":"Bearer "+self.api_key},
            json=self._chat_payload([
                {"role":"system","content":system_prompt},
                {"role":"user","content":user_prompt},
            ]),timeout=self.timeout_seconds,
        )
        content=self._content(response)
        if content.startswith("```"):
            content=content.split("\n",1)[1].rsplit("```",1)[0]
        return json.loads(content)

    def _request(self, **kwargs):
        """统一真实 provider 边界，并把可恢复故障映射为 Dispatcher 可识别的异常。"""

        try:
            response = self._post(**kwargs)
            response.raise_for_status()
            return response
        except Exception as exc:
            mapped = _retryable_provider_error(exc)
            if mapped is None or mapped is exc:
                raise
            raise mapped from exc

    @staticmethod
    def _content(response):
        try:
            content = response.json()["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise TypeError("message.content 必须是字符串")
            return content.strip()
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise MalformedLLMResponseError(
                "LLM provider 响应不符合 choices[0].message.content 协议"
            ) from exc

    @staticmethod
    def _post(**kwargs):
        """延迟加载 provider SDK，使无 LLM 的 Demo/Fake 路径不受其影响。"""
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError(
                "真实 LLM 路径需要安装 httpx；请安装项目依赖或清空 LLM 配置"
            ) from exc
        return httpx.post(**kwargs)

    def _chat_payload(self, messages, temperature=None):
        payload = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature if temperature is None else float(temperature),
        }
        if self.max_tokens > 0:
            payload["max_tokens"] = self.max_tokens
        if self.thinking_type:
            payload["thinking"] = {"type": self.thinking_type}
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        return payload


def _retryable_provider_error(exc):
    """在不把 provider SDK 引入领域层的前提下识别传输层瞬时故障。"""

    if isinstance(exc, LLMTimeoutError):
        return exc
    if isinstance(exc, TimeoutError):
        return LLMTimeoutError("LLM provider 请求超时")
    if isinstance(exc, ConnectionError):
        return exc

    error_name = type(exc).__name__
    if "Timeout" in error_name:
        return LLMTimeoutError("LLM provider 请求超时")
    if error_name in {
        "ConnectError",
        "NetworkError",
        "ReadError",
        "WriteError",
        "RemoteProtocolError",
        "PoolTimeout",
    }:
        return ConnectionError("LLM provider 网络或连接故障")

    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code is None:
        status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and (
        status_code in {408, 409, 425, 429} or status_code >= 500
    ):
        return ConnectionError(
            "LLM provider 暂时不可用（HTTP {}）".format(status_code)
        )
    if status_code in {401, 403}:
        return PermissionError(
            "LLM provider 拒绝访问（HTTP {}）：请检查 API Key 是否为方舟 API Key、"
            "账号是否已开通/订阅当前模型、endpoint 与 model id 是否匹配".format(
                status_code
            )
        )
    return None


def configured_llm():
    backend = configured_llm_backend()
    if backend == "fake":
        # FakeLLM is opt-in and remains isolated from the real provider path.
        from prievo_agent.infrastructure.local.fake_llm import FakeLLM

        return FakeLLM()
    endpoint=os.getenv("LLM_API_ENDPOINT","").strip()
    api_key=os.getenv("LLM_API_KEY","").strip() or os.getenv("ARK_API_KEY","").strip()
    model=os.getenv("LLM_MODEL","").strip()
    if endpoint and api_key and model:
        return OpenAICompatibleLLM(endpoint,api_key,model)
    return None


def configured_llm_backend():
    backend = os.getenv("LLM_BACKEND", "openai-compatible").strip().lower()
    if backend not in LLM_BACKENDS:
        raise ValueError(
            "LLM_BACKEND 必须是以下值之一：{}".format(
                ", ".join(sorted(LLM_BACKENDS))
            )
        )
    return backend


def _env_float(name, default):
    value = os.getenv(name, "").strip()
    if not value:
        return float(default)
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError("{} 必须是数字".format(name)) from exc
    if parsed <= 0:
        raise ValueError("{} 必须大于 0".format(name))
    return parsed


def _env_int(name, default):
    value = os.getenv(name, "").strip()
    if not value:
        return int(default)
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError("{} 必须是整数".format(name)) from exc
    if parsed < 0:
        raise ValueError("{} 必须大于等于 0".format(name))
    return parsed
