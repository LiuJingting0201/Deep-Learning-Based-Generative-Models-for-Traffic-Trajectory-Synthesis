"""Import smoke tests for the package skeleton."""


def test_package_imports() -> None:
    import cnr_trajectory

    assert cnr_trajectory.__version__ == "0.1.0"
