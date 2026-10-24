"""Sanity checks on the raw dataset. Run with:  pytest"""

from dropout import config
from dropout.data import decode, file_sha256, load_codebook, load_data


def test_raw_file_unchanged():
    assert file_sha256(config.RAW_DATA_PATH) == config.RAW_DATA_SHA256


def test_shape():
    df = load_data()
    assert df.shape == (4424, 37)  # 36 features + target


def test_no_missing_or_duplicates():
    df = load_data()
    assert df.isna().sum().sum() == 0
    assert df.duplicated().sum() == 0


def test_class_counts():
    counts = load_data()[config.TARGET].value_counts().to_dict()
    assert counts == {"Graduate": 2209, "Dropout": 1421, "Enrolled": 794}


def test_stage_feature_groups():
    assert len(config.ENROLLMENT_FEATURES) == 24
    assert len(config.SEM1_FEATURES) == 6
    assert len(config.SEM2_FEATURES) == 6
    assert len(config.STAGES["semester_2"]) == 36
    assert set(config.CATEGORICAL_FEATURES) <= set(config.ENROLLMENT_FEATURES)


def test_codebook_covers_every_code():
    df = load_data()
    codebook = load_codebook()
    for variable, rows in codebook.groupby("variable"):
        missing = set(df[variable].unique()) - set(rows["code"])
        assert not missing, f"{variable} has codes with no meaning: {missing}"


def test_decode():
    decoded = decode(load_data().head(2), ["course"])
    assert list(decoded["course"]) == ["Animation and Multimedia Design", "Tourism"]
