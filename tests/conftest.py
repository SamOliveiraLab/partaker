def pytest_configure(config):
    config.addinivalue_line("markers", "slow: requires model download / longer run")
