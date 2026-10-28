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

from __future__ import annotations

from typing import Optional

from nemoguardrails.manifests.config_schema import (
    Field,
    RailConfigBaseModel,
    RailConfigSpec,
    rail_field,
)


class NaceRailConfig(RailConfigBaseModel):
    """Defaults for the Drex rail. The API key stays in NACE_API_KEY."""

    model: Optional[str] = Field(
        default=None,
        description="Drex model id. Omitted calls use drex-v2.0. Unknown ids are forwarded. Images require drex-max.",
    )
    fail_at: float = Field(
        default=0.5,
        description="Block when the matching score or probability reaches this value.",
    )


def build_config_spec() -> RailConfigSpec:
    return RailConfigSpec(
        annotation=Optional[NaceRailConfig],
        field_info=rail_field(
            default_factory=NaceRailConfig,
            description="Configuration for the Drex rail.",
        ),
        exports={"NaceRailConfig": NaceRailConfig},
    )
