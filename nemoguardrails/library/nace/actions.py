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

"""NeMo actions for the Drex rail.

nace-sdk is imported inside the action so catalog discovery still works when
the package is absent.
"""

import json
from typing import Any, Dict, Optional

from nemoguardrails.actions import action
from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.guardrails.tool_schema import tool_call_validation


def _defaults(config: Any) -> tuple[Optional[str], float]:
    rails = getattr(config, "rails", None)
    data = getattr(rails, "config", None)
    nace = getattr(data, "nace", None)
    if nace is None:
        return None, 0.5
    fail_at = getattr(nace, "fail_at", 0.5)
    return getattr(nace, "model", None), 0.5 if fail_at is None else fail_at


def _outcome(result: Any) -> RailOutcome:
    if result.is_blocked:
        return RailOutcome.block(metadata=result.metadata)
    return RailOutcome.allow(metadata=result.metadata)


def tool_call_text(tool_call: Any) -> str:
    function = getattr(tool_call, "function", None)
    name = getattr(function, "name", None) or getattr(tool_call, "type", "tool")
    arguments = getattr(function, "arguments", None)
    return json.dumps({"tool": name, "arguments": arguments})


def _load():
    try:
        from nace_sdk import NaceAPIError

        from nemoguardrails.library.nace.check import CheckError, open_client, prepare, questions_for, run_check
    except ImportError as exc:
        raise ImportError("The Drex rail requires nace-sdk. Install it with `pip install nace-sdk`.") from exc
    return NaceAPIError, CheckError, open_client, prepare, questions_for, run_check


@action(name="DrexCheckAction", is_system_action=True)
async def drex_check(
    text: str,
    questions: Optional[Dict[str, Any]] = None,
    question: Optional[str] = None,
    model: Optional[str] = None,
    images: Optional[list] = None,
    fail_at: Optional[float] = None,
    fail_when: Optional[str] = None,
    blocked_choices: Optional[Dict[str, str]] = None,
    config: Any = None,
    **_kwargs: Any,
) -> RailOutcome:
    """Check text, and images on drex-max, against Drex questions."""
    NaceAPIError, CheckError, open_client, prepare, questions_for, run_check = _load()
    configured_model, configured_fail_at = _defaults(config)
    chosen_model = configured_model if model is None else model
    chosen_fail_at = configured_fail_at if fail_at is None else fail_at
    client = None
    try:
        built = questions_for(questions, question)
        prepare(text, built, model=chosen_model, images=images)
        client = open_client()
        result = await run_check(
            client,
            text,
            questions,
            question=question,
            model=chosen_model,
            images=images,
            fail_at=chosen_fail_at,
            fail_when=fail_when,
            blocked_choices=blocked_choices,
        )
    except (CheckError, NaceAPIError) as exc:
        return RailOutcome.failure(reason=str(exc))
    finally:
        if client is not None:
            await client.aclose()
    return _outcome(result)


@action(name="DrexToolCheckAction", is_system_action=True)
@tool_call_validation
async def drex_check_tool_call(
    tool_call: Any,
    tool_definition: Any = None,
    questions: Optional[Dict[str, Any]] = None,
    question: Optional[str] = None,
    model: Optional[str] = None,
    fail_at: Optional[float] = None,
    config: Any = None,
    **_kwargs: Any,
) -> RailOutcome:
    """Check a tool call's arguments before it runs."""
    del tool_definition
    return await drex_check(
        tool_call_text(tool_call),
        questions=questions,
        question=question,
        model=model,
        fail_at=fail_at,
        config=config,
    )
