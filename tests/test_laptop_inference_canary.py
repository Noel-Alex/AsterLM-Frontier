from scripts.run_laptop_inference_canary import (
    EXPECTED_ACTIVE_PARAMETERS,
    EXPECTED_LAYERS,
    EXPECTED_TOTAL_PARAMETERS,
    validate_measurement,
)


def _result() -> dict:
    return {
        "status": "passed",
        "architecture": {
            "effective_parameters": EXPECTED_TOTAL_PARAMETERS,
            "active_parameters_estimate": EXPECTED_ACTIVE_PARAMETERS,
            "n_layers": EXPECTED_LAYERS,
        },
        "runtime": {
            "executed_block_count": EXPECTED_LAYERS,
            "decoded_tokens": 4,
            "cache_bytes": 1024,
            "prefill_tokens_per_second": 100.0,
            "decode_tokens_per_second": 10.0,
            "peak_allocated_gib": 4.0,
            "finite_logits": True,
        },
    }


def test_measurement_requires_exact_selected_architecture_and_cached_decode() -> None:
    validate_measurement(_result())

    wrong = _result()
    wrong["architecture"]["effective_parameters"] -= 1
    try:
        validate_measurement(wrong)
    except ValueError as exc:
        assert "wrong total parameter" in str(exc)
    else:
        raise AssertionError("wrong architecture was accepted")


def test_measurement_rejects_non_finite_logits() -> None:
    result = _result()
    result["runtime"]["finite_logits"] = False
    try:
        validate_measurement(result)
    except ValueError as exc:
        assert "non-finite" in str(exc)
    else:
        raise AssertionError("non-finite inference was accepted")
