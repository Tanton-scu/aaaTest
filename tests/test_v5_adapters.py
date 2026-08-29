import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prievo_agent.domain.models import Event
from prievo_agent.infrastructure.llm_adapter import configured_llm
from prievo_agent.infrastructure.llm_adapter import OpenAICompatibleLLM
from prievo_agent.infrastructure.llm_adapter import MalformedLLMResponseError
from prievo_agent.infrastructure.local_runtime import LocalRuntimeComposition
from prievo_agent.infrastructure.env_loader import load_env_file
from prievo_agent.infrastructure.redis_events import PublishingStore
from prievo_agent.agents.evolution_planner import EvolutionPlannerAgent


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": (
            '{"code":"def run_tuners(file, budget, seed, maxlives):\\n'
            '    return 1","description":"real adapter","operators":["Revise"]}'
        )}}]}


class FakeHTTPStatusError(RuntimeError):
    def __init__(self, status_code):
        self.response = type("Response", (), {"status_code": status_code})()
        super().__init__("HTTP {}".format(status_code))


class FakeStore:
    def __init__(self):
        self.persisted = []

    def append_event(self, run_id, event_type, message, **payload):
        event = Event(1, run_id, event_type, message, payload)
        self.persisted.append(event)
        return event


class FailingBus:
    def publish(self, run_id, event):
        return False


class V5AdapterTest(unittest.TestCase):
    def test_openai_compatible_llm_is_real_configurable_path(self):
        environment = {"LLM_API_ENDPOINT": "https://llm.example/v1/chat/completions",
                       "LLM_API_KEY": "secret", "LLM_MODEL": "model"}
        with patch.dict(os.environ, environment, clear=False), patch(
                "prievo_agent.infrastructure.llm_adapter.OpenAICompatibleLLM._post",
                return_value=FakeResponse()) as request:
            candidate = configured_llm().generate_candidate("m1", [], 2)
        self.assertEqual("real adapter", candidate.description)
        self.assertEqual(["Revise"], candidate.operators)
        self.assertEqual("Bearer secret", request.call_args.kwargs["headers"]["Authorization"])
        self.assertEqual(
            "https://llm.example/v1/chat/completions",
            request.call_args.kwargs["url"],
        )

    def test_real_agent_json_adapter_uses_keyword_url_contract(self):
        llm = OpenAICompatibleLLM(
            "https://llm.example/v1/chat/completions", "secret", "model"
        )
        response = FakeResponse()
        response.json = lambda: {
            "choices": [{"message": {"content": '{"selected_instance_ids":["A"]}'}}]
        }
        with patch.object(llm, "_post", return_value=response) as request:
            result = llm.generate_similarity_decision("prompt", ["A"])
        self.assertEqual(["A"], result["selected_instance_ids"])
        self.assertEqual(
            "https://llm.example/v1/chat/completions",
            request.call_args.kwargs["url"],
        )
        self.assertEqual((), request.call_args.args)

    def test_transient_provider_status_is_mapped_for_bounded_redrive(self):
        llm = OpenAICompatibleLLM("https://llm.example", "secret", "model")
        response = FakeResponse()
        response.raise_for_status = lambda: (_ for _ in ()).throw(
            FakeHTTPStatusError(429)
        )
        with patch.object(llm, "_post", return_value=response):
            with self.assertRaises(ConnectionError):
                llm.generate_similarity_decision("prompt", ["A"])

    def test_auth_provider_status_is_reported_as_permission_error(self):
        llm = OpenAICompatibleLLM("https://llm.example", "secret", "model")
        response = FakeResponse()
        response.raise_for_status = lambda: (_ for _ in ()).throw(
            FakeHTTPStatusError(403)
        )
        with patch.object(llm, "_post", return_value=response):
            with self.assertRaisesRegex(PermissionError, "API Key"):
                llm.generate_similarity_decision("prompt", ["A"])

    def test_malformed_provider_envelope_is_a_bounded_protocol_error(self):
        llm = OpenAICompatibleLLM("https://llm.example", "secret", "model")
        response = FakeResponse()
        response.json = lambda: {"choices": []}
        with patch.object(llm, "_post", return_value=response):
            with self.assertRaises(MalformedLLMResponseError):
                llm.generate_similarity_decision("prompt", ["A"])

    def test_redis_notification_failure_does_not_rollback_durable_event(self):
        inner = FakeStore()
        event = PublishingStore(inner, FailingBus()).append_event(
            "run-1", "GENERATION_STARTED", "开始")
        self.assertEqual(1, event.sequence)
        self.assertEqual(1, len(inner.persisted))

    def test_full_mode_refuses_non_mysql_database(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                os.environ, {"PRIEVO_MODE": "full", "DATABASE_URL": "sqlite:///bad.db",
                             "REDIS_URL": ""}, clear=False):
            with self.assertRaisesRegex(RuntimeError, r"mysql\+pymysql"):
                LocalRuntimeComposition(Path(directory))

    def test_partial_llm_configuration_fails_instead_of_silently_using_fake(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                os.environ,
                {"PRIEVO_MODE": "demo", "LLM_API_ENDPOINT": "https://llm.example",
                 "LLM_API_KEY": "", "LLM_MODEL": ""},
                clear=False):
            with self.assertRaisesRegex(RuntimeError, "LLM 配置不完整"):
                LocalRuntimeComposition(Path(directory))

    def test_ark_api_key_counts_as_complete_llm_configuration(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                os.environ,
                {"PRIEVO_MODE": "demo", "LLM_API_ENDPOINT": "https://llm.example",
                 "LLM_API_KEY": "", "ARK_API_KEY": "ark-fixture",
                 "LLM_MODEL": "glm-5-2-260617"},
                clear=False):
            composition = LocalRuntimeComposition(Path(directory))
            snapshot = composition.health_snapshot()
        self.assertEqual("openai-compatible", snapshot["llm"])

    def test_load_env_file_does_not_override_existing_environment(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                os.environ, {"LLM_MODEL": "from-shell"}, clear=False):
            path = Path(directory) / ".env"
            path.write_text(
                "LLM_MODEL=from-file\nARK_API_KEY='hidden-fixture'\n",
                encoding="utf-8",
            )
            loaded = load_env_file(path)
            self.assertEqual("from-shell", os.environ["LLM_MODEL"])
            self.assertEqual("hidden-fixture", os.environ["ARK_API_KEY"])
            self.assertNotIn("LLM_MODEL", loaded)

    def test_planner_normalizes_redundant_parent_count_from_real_llm(self):
        plan = EvolutionPlannerAgent._parse_plan(
            {
                "generation_strategy": "i1",
                "parent_selection_policy": "roulette",
                "decision_reason": "real model preferred exploration",
                "required_parent_count": 2,
            },
            run_id="run-1",
            generation=1,
            sequence=0,
            scheduled_strategy="i1",
            context_artifact_id="ctx",
            prompt_artifact_id="prompt",
            previous_feedback=None,
        )
        self.assertEqual("i1", plan.generation_strategy)
        self.assertEqual("null", plan.parent_selection_policy)
        self.assertEqual(0, plan.required_parent_count)
        self.assertIn("Backend contract normalization", plan.decision_reason)

    def test_planner_normalizes_null_policy_for_parent_strategy(self):
        plan = EvolutionPlannerAgent._parse_plan(
            {
                "generation_strategy": "e1",
                "parent_selection_policy": "null",
                "decision_reason": "real model answered with redundant fields",
                "required_parent_count": 0,
            },
            run_id="run-1",
            generation=1,
            sequence=0,
            scheduled_strategy="e1",
            context_artifact_id="ctx",
            prompt_artifact_id="prompt",
            previous_feedback=None,
        )
        self.assertEqual("roulette", plan.parent_selection_policy)
        self.assertEqual(2, plan.required_parent_count)

    def test_health_snapshot_explains_fake_llm_and_non_fact_redis(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                os.environ,
                {"PRIEVO_MODE": "demo", "LLM_API_ENDPOINT": "",
                 "LLM_API_KEY": "", "LLM_MODEL": ""},
                clear=False):
            composition = LocalRuntimeComposition(Path(directory))
            snapshot = composition.health_snapshot()
        self.assertEqual("sqlite", snapshot["database"])
        self.assertEqual("disabled", snapshot["redis"])
        self.assertEqual("deterministic-fake", snapshot["llm"])
        self.assertEqual("inline", snapshot["evaluation_execution_mode"])
        self.assertEqual(10.0, snapshot["evaluation_timeout_seconds"])

    def test_evaluation_timeout_environment_is_validated(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                os.environ,
                {"PRIEVO_MODE": "demo", "EVALUATION_TIMEOUT_SECONDS": "0"},
                clear=False):
            with self.assertRaisesRegex(ValueError, "EVALUATION_TIMEOUT_SECONDS"):
                LocalRuntimeComposition(Path(directory))


if __name__ == "__main__":
    unittest.main()
