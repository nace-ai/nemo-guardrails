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

from nemoguardrails.manifests import (
    ActionRef,
    Binding,
    ConfigSpecRef,
    EnvVar,
    RailActions,
    RailConfigSchema,
    RailDirection,
    RailFlows,
    RailManifest,
    RailMetadata,
    RailPrivacy,
    RailRequirements,
    RailSpec,
    RailSurface,
    ServiceRequirement,
)

DREX_CHECK = ActionRef(
    name="DrexCheckAction",
    target="nemoguardrails.library.nace.actions:drex_check",
)
DREX_TOOL_CHECK = ActionRef(
    name="DrexToolCheckAction",
    target="nemoguardrails.library.nace.actions:drex_check_tool_call",
)

RAIL = RailManifest(
    name="nace",
    metadata=RailMetadata(
        display_name="Drex",
        description="Checks text and drex-max images with Drex. Text defaults to drex-v2.0.",
        categories=("input", "output", "retrieval", "tool_call"),
        capabilities=("allow", "block", "classify"),
        tags=("third-party", "api", "policy"),
        docs_url="docs/configure-rails/guardrail-catalog/community/drex.mdx",
    ),
    spec=RailSpec(
        config_schema=RailConfigSchema(
            key="nace",
            spec=ConfigSpecRef(target="nemoguardrails.library.nace.rail_config:build_config_spec"),
        ),
        flows=RailFlows(
            flow_names=(
                "drex check input",
                "drex check output",
                "drex check retrieval",
                "drex check tool call",
            ),
        ),
        actions=RailActions(refs=(DREX_CHECK, DREX_TOOL_CHECK)),
        surfaces=(
            RailSurface(
                name="drex check input",
                direction=RailDirection.INPUT,
                action=DREX_CHECK,
                bindings=(
                    Binding.context("text", "user_message"),
                    Binding.surface_param("question", "question"),
                    Binding.surface_param("model", "model", required=False),
                ),
            ),
            RailSurface(
                name="drex check output",
                direction=RailDirection.OUTPUT,
                action=DREX_CHECK,
                bindings=(
                    Binding.context("text", "bot_message"),
                    Binding.surface_param("question", "question"),
                    Binding.surface_param("model", "model", required=False),
                ),
            ),
            RailSurface(
                name="drex check retrieval",
                direction=RailDirection.RETRIEVAL,
                action=DREX_CHECK,
                bindings=(
                    Binding.context("text", "relevant_chunks"),
                    Binding.surface_param("question", "question"),
                    Binding.surface_param("model", "model", required=False),
                ),
            ),
            RailSurface(
                name="drex check tool call",
                direction=RailDirection.TOOL_CALL,
                action=DREX_TOOL_CHECK,
                bindings=(
                    Binding.context("tool_call", "tool_call"),
                    Binding.context("tool_definition", "tool_definition"),
                    Binding.surface_param("question", "question"),
                    Binding.surface_param("model", "model", required=False),
                ),
            ),
        ),
        requirements=RailRequirements(
            env_vars=(EnvVar(name="NACE_API_KEY", required=True),),
            services=(ServiceRequirement(name="Drex", required=True),),
            optional_dependencies=("nace-sdk",),
        ),
        privacy=RailPrivacy(
            sends_user_text=True,
            sends_bot_text=True,
            sends_retrieved_chunks=True,
            sends_tool_call_text=True,
            remote_services=("Drex",),
        ),
    ),
)
