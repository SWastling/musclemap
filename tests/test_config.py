import musclemap.config as config


def test_config():
    assert isinstance(config.FSL_VERSIONS, list)
    assert isinstance(config.FSL_VERSIONS[0], str)
