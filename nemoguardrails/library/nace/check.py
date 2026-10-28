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

"""Drex check used by the NeMo Guardrails action.

The NeMo decorator stays in actions.py. This module is the part tests can run
without installing NeMo.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from nace_sdk import AsyncNaceClient, Choice, Noul, RetryPolicy, Score

VERSION = "0.1.0"
CLIENT_ID = f"nemo-guardrails-nace/{VERSION}"
DEFAULT_MODEL = "drex-v2.0"
IMAGE_MODEL = "drex-max"
MAX_QUESTIONS = {"drex-max": 16, "drex-v2.0": 512}
MAX_IMAGES = 8
CALL_TIMEOUT_SECONDS = 60
RETRY_BUDGET_SECONDS = 180
QUESTION_TYPES = {"noul": Noul, "choice": Choice, "score": Score}


class CheckError(Exception):
    """The check was not sent. This is not a blocked rail."""


@dataclass(frozen=True)
class CheckOutcome:
    is_blocked: bool
    model: str
    metadata: dict[str, Any]


def client_settings() -> dict[str, Any]:
    return {
        "timeout": CALL_TIMEOUT_SECONDS,
        "retry": RetryPolicy(max_retries=8, timeout=RETRY_BUDGET_SECONDS),
        "client_info": CLIENT_ID,
    }


def open_client(*, api_key: str | None = None, base_url: str | None = None) -> AsyncNaceClient:
    settings = client_settings()
    return AsyncNaceClient(
        api_key=api_key,
        base_url=base_url,
        timeout=settings["timeout"],
        retry=settings["retry"],
        client_info=settings["client_info"],
    )


def model_name(model: str | None) -> str:
    if model is None:
        return DEFAULT_MODEL
    if not isinstance(model, str) or not model.strip():
        raise CheckError("model must be a non-empty string")
    return model


def as_question(question: Any) -> Noul | Choice | Score:
    if isinstance(question, str):
        raise CheckError("a free-form string is not a Drex question")
    if isinstance(question, (Noul, Choice, Score)):
        return question
    if isinstance(question, Mapping):
        kind = question.get("type")
        cls = QUESTION_TYPES.get(kind) if isinstance(kind, str) else None
        if cls is None:
            raise CheckError("question type must be noul, choice, or score")
        try:
            return cls.model_validate(dict(question))
        except Exception as exc:
            raise CheckError("question dict did not match a Drex question") from exc
    raise CheckError("question must be a nace_sdk question or a question dict")


def prepare(
    text: str,
    questions: Mapping[str, Any],
    *,
    model: str | None = None,
    images: Sequence[Any] | None = None,
) -> tuple[str, dict[str, Noul | Choice | Score], tuple[Any, ...]]:
    if not isinstance(text, str) or not text:
        raise CheckError("text must be a non-empty string")
    chosen = model_name(model)
    if not isinstance(questions, Mapping) or not questions:
        raise CheckError("questions must be a non-empty mapping")
    built: dict[str, Noul | Choice | Score] = {}
    for name, question in questions.items():
        if not isinstance(name, str) or not name:
            raise CheckError("question names must be non-empty strings")
        built[name] = as_question(question)
    cap = MAX_QUESTIONS.get(chosen)
    if cap is not None and len(built) > cap:
        raise CheckError(f"{chosen} accepts at most {cap} questions")
    attached = tuple(images or ())
    if attached:
        if chosen != IMAGE_MODEL:
            raise CheckError(f"images require model {IMAGE_MODEL!r}, not {chosen!r}")
        if len(attached) > MAX_IMAGES:
            raise CheckError(f"{IMAGE_MODEL} accepts at most {MAX_IMAGES} images")
    return chosen, built, attached


def _blocks(
    name: str,
    question: Noul | Choice | Score,
    answer: Any,
    fail_at: float,
    fail_when: str | None,
    blocked_choices: Mapping[str, str],
) -> bool:
    if isinstance(question, Noul):
        side = "yes" if fail_when is None else fail_when
        probability = float(answer.noul) if side == "yes" else 1.0 - float(answer.noul)
        return probability >= fail_at
    if isinstance(question, Score):
        score = float(answer.score)
        if fail_when == "below":
            return score < fail_at
        return score >= fail_at
    blocked = blocked_choices.get(name)
    if blocked is None:
        return False
    return answer.choice == blocked and float(answer.confidence) >= fail_at


def outcome_from(
    response: Any,
    questions: Mapping[str, Noul | Choice | Score],
    model: str,
    *,
    fail_at: float = 0.5,
    fail_when: str | None = None,
    blocked_choices: Mapping[str, str] | None = None,
) -> CheckOutcome:
    answers = getattr(response, "answers", None) or {}
    blocked_on = dict(blocked_choices or {})
    blocked = False
    reported: dict[str, Any] = {}
    for name, question in questions.items():
        answer = answers.get(name)
        if answer is None:
            raise CheckError(f"response has no answer for {name!r}")
        reported[name] = answer.model_dump() if hasattr(answer, "model_dump") else dict(answer)
        blocked = blocked or _blocks(name, question, answer, fail_at, fail_when, blocked_on)
    return CheckOutcome(
        is_blocked=blocked,
        model=getattr(response, "model", None) or model,
        metadata={"answers": reported, "request_id": getattr(response, "request_id", None)},
    )


def questions_for(questions: Mapping[str, Any] | None, question: str | None) -> Mapping[str, Any]:
    if questions:
        return questions
    if isinstance(question, str) and question.strip():
        return {"check": Noul(instructions=question)}
    raise CheckError("questions must be a non-empty mapping, or question must be a non-empty string")


async def run_check(
    client: Any,
    text: str,
    questions: Mapping[str, Any] | None = None,
    *,
    question: str | None = None,
    model: str | None = None,
    images: Sequence[Any] | None = None,
    fail_at: float = 0.5,
    fail_when: str | None = None,
    blocked_choices: Mapping[str, str] | None = None,
) -> CheckOutcome:
    built_input = questions_for(questions, question)
    chosen, built, attached = prepare(text, built_input, model=model, images=images)
    response = await client.system_one(
        text,
        built,
        model=chosen,
        images=list(attached) if attached else None,
    )
    return outcome_from(
        response,
        built,
        chosen,
        fail_at=fail_at,
        fail_when=fail_when,
        blocked_choices=blocked_choices,
    )
