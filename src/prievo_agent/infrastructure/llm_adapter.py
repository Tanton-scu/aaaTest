import json
import os

from prievo_agent.core.models import GeneratedCandidate
from prievo_agent.domain.errors import LLMTimeoutError


class MalformedLLMResponseError(ValueError):
    """Provider 返回成功响应，但响应体不符合约定的结构化协议。"""


class OpenAICompatibleLLM:
    def __init__(self,endpoint,api_key,model):
        self.endpoint=endpoint.rstrip("/")
        self.api_key=api_key
        self.model=model

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
            json={"model":self.model,"messages":[
                {"role":"system","content":"Generate safe, deterministic heuristic source code. Output JSON only."},
                {"role":"user","content":prompt}],"temperature":0.3},timeout=90,
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
            json={"model":self.model,"messages":[
                {"role":"system","content":system_prompt},
                {"role":"user","content":user_prompt}],"temperature":0.1},timeout=90,
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
    return None


def configured_llm():
    endpoint=os.getenv("LLM_API_ENDPOINT","").strip()
    api_key=os.getenv("LLM_API_KEY","").strip()
    model=os.getenv("LLM_MODEL","").strip()
    if endpoint and api_key and model:
        return OpenAICompatibleLLM(endpoint,api_key,model)
    return None
