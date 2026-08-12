from __future__ import annotations

from asterlm.cuda_toolchain import _major_minor


def test_cuda_package_version_parser_handles_pep440_build_versions():
    assert _major_minor("13.3.3.4.1") == (13, 3)
    assert _major_minor("2.13.0+cu130") == (2, 13)
    assert _major_minor(None) is None
