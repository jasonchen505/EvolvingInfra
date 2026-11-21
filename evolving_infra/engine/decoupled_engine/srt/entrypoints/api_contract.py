# Adapted from SGLang python/sglang/srt/entrypoints/api_contract.py (commit 297e54fd10), modified for evolving-infra.
"""The HTTP JSON contract gate: `proto/sglang/api/v1` is the schema for both
the Rust and the Python server, so a `/generate` body the Python dataclass
parse admitted but the schema rejects is refused here with the schema's own
error text. The generated decoder is the single source of that judgment.

The schema is SGLang's. It ignores unknown request fields but refuses unknown
sampling parameters, so the sampling parameters this engine adds are taken out
before the check: `SamplingParams` has already admitted them.
"""

from typing import Any, Optional

from decoupled_engine.api.v1.api_types import GenerateRequest, JsonContractError

# The field of SamplingParams that SGLang's schema does not have (strict replay).
SAMPLING_EXTENSION_FIELDS = frozenset({"forced_output_token_ids"})


def _without_extensions(body: Any) -> Any:
    """`body` without this engine's own sampling parameters, in one
    `sampling_params` object or in a list of them; any other shape as is."""

    def strip(sampling: Any) -> Any:
        if not isinstance(sampling, dict):
            return sampling
        return {
            k: v for k, v in sampling.items() if k not in SAMPLING_EXTENSION_FIELDS
        }

    if not isinstance(body, dict) or "sampling_params" not in body:
        return body
    sampling = body["sampling_params"]
    if isinstance(sampling, list):
        return {**body, "sampling_params": [strip(item) for item in sampling]}
    return {**body, "sampling_params": strip(sampling)}


def generate_contract_error(body: Any) -> Optional[str]:
    """The contract violation in a parsed `/generate` JSON body, or None."""
    try:
        GenerateRequest.from_json_value(_without_extensions(body))
    except JsonContractError as e:
        return str(e)
    return None
