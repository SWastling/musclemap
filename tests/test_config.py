import musclemap.config as config


def test_config():

    assert isinstance(config.NIBABEL_VERSIONS, list)
    assert isinstance(config.NIBABEL_VERSIONS[0], str)
    assert isinstance(config.FSL_VERSIONS, list)
    assert isinstance(config.FSL_VERSIONS[0], str)


