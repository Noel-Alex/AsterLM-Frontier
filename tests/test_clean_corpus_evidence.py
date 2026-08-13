from scripts.import_clean_corpus_evidence import DEFAULT_SOURCES


def test_default_clean_sources_include_math_but_not_unavailable_code() -> None:
    assert set(DEFAULT_SOURCES) == {
        "fineweb_edu",
        "dclm",
        "cosmopedia_v2",
        "finemath_4plus",
        "nemotron_math",
    }
    assert all("code" not in source for source in DEFAULT_SOURCES)
