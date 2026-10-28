# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
from types import SimpleNamespace

import pytest
from nace_sdk import Choice, ChoiceAnswer, Noul, NoulAnswer, Score, ScoreAnswer

from nemoguardrails.library.nace.actions import drex_check, drex_check_tool_call, tool_call_text
from nemoguardrails.library.nace.check import (
    CALL_TIMEOUT_SECONDS,
    DEFAULT_MODEL,
    IMAGE_MODEL,
    RETRY_BUDGET_SECONDS,
    CheckError,
    client_settings,
    prepare,
    run_check,
)
from nemoguardrails.rails.llm.config import RailsConfig


def test_a_question_string_becomes_one_noul() -> None:
    model, questions, _ = prepare("note", {"check": Noul(instructions="Is this a violation?")})
    assert model == "drex-v2.0"
    assert questions["check"].instructions == "Is this a violation?"


def test_run_check_accepts_a_question_string() -> None:
    class Client:
        async def system_one(self, _text, questions, *, model, images):
            assert set(questions) == {"check"}
            assert model == "drex-v2.0"
            return SimpleNamespace(model=model, request_id=None, answers={"check": NoulAnswer(noul=0.1)})

    outcome = asyncio.run(run_check(Client(), "note", question="Is this a violation?"))
    assert outcome.is_blocked is False


def test_omitted_model_is_drex_v2() -> None:
    model, questions, images = prepare("note", {"ok": Noul(instructions="Is this fine?")})
    assert model == DEFAULT_MODEL == "drex-v2.0"
    assert images == ()
    assert set(questions) == {"ok"}


def test_unknown_model_is_forwarded() -> None:
    model, _, _ = prepare("note", {"ok": Noul(instructions="Is this fine?")}, model="drex-next")
    assert model == "drex-next"


def test_images_are_refused_before_a_text_model_call() -> None:
    with pytest.raises(CheckError, match="drex-max"):
        prepare("note", {"ok": Noul(instructions="Is this fine?")}, model="drex-v2.0", images=["red"])


def test_drex_max_rejects_more_than_sixteen_questions() -> None:
    questions = {f"q{i}": Noul(instructions="Is this fine?") for i in range(17)}
    with pytest.raises(CheckError, match="16"):
        prepare("note", questions, model=IMAGE_MODEL)


def test_client_budget_covers_the_evaluate_timeout_and_retries() -> None:
    settings = client_settings()
    assert settings["timeout"] == CALL_TIMEOUT_SECONDS >= 60
    assert settings["retry"].timeout == RETRY_BUDGET_SECONDS == 180
    assert 429 in settings["retry"].http_statuses
    assert 529 in settings["retry"].http_statuses
    assert settings["client_info"] == "nemo-guardrails-nace/0.1.0"


def test_run_check_does_not_call_when_images_are_on_the_wrong_model() -> None:
    class Client:
        def __init__(self) -> None:
            self.calls = 0

        async def system_one(self, *_args, **_kwargs):
            self.calls += 1

    client = Client()
    with pytest.raises(CheckError):
        asyncio.run(
            run_check(
                client,
                "note",
                {"seen": Noul(instructions="Is an image present?")},
                model="drex-v2.0",
                images=["red"],
            )
        )
    assert client.calls == 0


def test_run_check_blocks_on_noul_and_keeps_the_probability() -> None:
    class Client:
        async def system_one(self, text, questions, *, model, images):
            assert text == "note"
            assert model == "drex-v2.0"
            assert images is None
            return SimpleNamespace(
                model=model,
                request_id="req-1",
                answers={"violation": NoulAnswer(noul=0.91)},
            )

    outcome = asyncio.run(
        run_check(Client(), "note", {"violation": Noul(instructions="Is this a violation?")}, fail_at=0.5)
    )
    assert outcome.is_blocked is True
    assert outcome.metadata["answers"]["violation"]["noul"] == 0.91
    assert outcome.metadata["request_id"] == "req-1"


def test_score_and_choice_use_the_same_call() -> None:
    class Client:
        async def system_one(self, _text, questions, *, model, images):
            assert model == "drex-max"
            assert images == ["red"]
            assert set(questions) == {"level", "route"}
            return SimpleNamespace(
                model=model,
                request_id=None,
                answers={
                    "level": ScoreAnswer(score=4, confidence=0.8, legend=("low", "high"), probabilities={4: 0.8}),
                    "route": ChoiceAnswer(choice="block", confidence=0.7, probabilities={"block": 0.7, "allow": 0.3}),
                },
            )

    outcome = asyncio.run(
        run_check(
            Client(),
            "note",
            {
                "level": Score(instructions="How severe?", criteria=["none", "low", "mid", "high", "severe"]),
                "route": Choice(instructions="Which route?", criteria={"block": "stop", "allow": "continue"}),
            },
            model="drex-max",
            images=["red"],
            fail_at=3,
            blocked_choices={"route": "block"},
        )
    )
    assert outcome.is_blocked is True


class _Client:
    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    async def system_one(self, text, questions, *, model, images):
        self.calls.append({"text": text, "model": model, "images": images, "questions": questions})
        return SimpleNamespace(model=model, request_id=None, answers=self.answers)

    async def aclose(self):
        return None


def _config(**nace):
    return RailsConfig.from_content(
        config={
            "models": [],
            "rails": {"config": {"nace": nace}},
        }
    )


def test_tool_call_text_keeps_the_name_and_arguments() -> None:
    tool_call = SimpleNamespace(function=SimpleNamespace(name="lookup", arguments={"q": "x"}), type="function")
    assert json.loads(tool_call_text(tool_call)) == {"tool": "lookup", "arguments": {"q": "x"}}


def test_check_error_becomes_a_failed_block(monkeypatch) -> None:
    from nemoguardrails.library.nace import check

    def refuse(**_kwargs):
        raise AssertionError("client should not open")

    monkeypatch.setattr(check, "open_client", refuse)
    outcome = asyncio.run(drex_check("", question="Is this a violation?"))
    assert outcome.is_blocked is True
    assert outcome.failed is True
    assert "non-empty" in outcome.reason


def test_api_error_becomes_a_failed_block(monkeypatch) -> None:
    import httpx
    from nace_sdk import NaceAPIError

    from nemoguardrails.library.nace import check

    class Client:
        async def system_one(self, *_args, **_kwargs):
            raise NaceAPIError(503, {"error": "down"}, httpx.Headers())

        async def aclose(self):
            return None

    monkeypatch.setattr(check, "open_client", lambda **_kwargs: Client())
    outcome = asyncio.run(drex_check("note", question="Is this a violation?"))
    assert outcome.failed is True
    assert outcome.is_blocked is True


def test_config_model_and_fail_at_apply_when_the_caller_omits_them(monkeypatch) -> None:
    from nemoguardrails.library.nace import check

    client = _Client({"check": NoulAnswer(noul=0.3)})
    monkeypatch.setattr(check, "open_client", lambda **_kwargs: client)
    outcome = asyncio.run(
        drex_check("note", question="Is this a violation?", config=_config(model="drex-next", fail_at=0.2))
    )
    assert client.calls[0]["model"] == "drex-next"
    assert outcome.is_blocked is True


def test_caller_model_wins_over_config(monkeypatch) -> None:
    from nemoguardrails.library.nace import check

    client = _Client({"check": NoulAnswer(noul=0.1)})
    monkeypatch.setattr(check, "open_client", lambda **_kwargs: client)
    asyncio.run(
        drex_check(
            "note",
            question="Is this a violation?",
            model="drex-v2.0",
            config=_config(model="drex-next"),
        )
    )
    assert client.calls[0]["model"] == "drex-v2.0"


def test_tool_check_failure_does_not_call_drex(monkeypatch) -> None:
    from nemoguardrails.guardrails.tool_schema import Tool
    from nemoguardrails.library.nace import check
    from nemoguardrails.types import ToolCall, ToolCallFunction

    def refuse(**_kwargs):
        raise AssertionError("client should not open")

    monkeypatch.setattr(check, "open_client", refuse)
    tool_call = ToolCall(id="call-1", function=ToolCallFunction(name="lookup", arguments={"q": "x"}))
    tool = Tool(
        name="lookup",
        arguments_schema={
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        },
    )
    outcome = asyncio.run(drex_check_tool_call(tool_call=tool_call, tool_definition=tool))
    assert outcome.failed is True
