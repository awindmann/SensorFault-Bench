from config_loader import load_dataset_windows


def test_shipped_dataset_windows_include_paper_batch_size_policy():
    defaults = {
        "DATA_FILES": ["ETTh1", "traffic", "Penmanshiel_Hourly_WT08", "BeijingAir_Tiantan"],
        "DATA_ROOT": "data/processed",
    }
    windows = load_dataset_windows(defaults=defaults)

    assert windows["ETTh1"]["batch_size"] == 16
    assert windows["traffic"]["batch_size"] == 16
    assert windows["Penmanshiel_Hourly_WT08"]["batch_size"] == 64
    assert windows["BeijingAir_Tiantan"]["batch_size"] == 64
