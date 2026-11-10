import io
import zipfile
from urllib.error import HTTPError

import numpy as np


def test_select_complete_rectus_femoris_task_pairs_only():
    from experiments.reference_fatigue.figshare_task_level import select_paired_rf_members

    names = [
        "Knee extension/S1/PersonA_30_RF_edited_final.mat",
        "Knee extension/S1/PersonA_TLim_RF_edited_final.mat",
        "Knee extension/S1/PersonA_30_VL_edited_final.mat",
        "Knee extension/S2/PersonB_30_RF_edited_final.mat",
        "Knee extension/S2/PersonB_TLim_VM_edited_final.mat",
        "Plantarflexion/S1/PersonA_30_RF_edited_final.mat",
    ]
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as handle:
        for name in names:
            handle.writestr(name, b"fixture")
    archive.seek(0)

    with zipfile.ZipFile(archive) as handle:
        pairs = select_paired_rf_members(handle.infolist())

    assert list(pairs) == ["S1"]
    assert set(pairs["S1"]) == {"30", "tlim"}
    assert all("_RF_" in info.filename for info in pairs["S1"].values())


def test_force_bout_detection_discards_short_post_task_mvc_bout():
    from experiments.reference_fatigue.figshare_task_level import detect_force_bouts

    force = np.concatenate([
        np.full(30, -1.0),
        np.full(50, 10.0),
        np.full(20, -1.0),
        np.full(50, 10.0),
        np.full(30, -1.0),
        np.full(10, 30.0),
        np.full(20, -1.0),
    ])

    bouts, threshold = detect_force_bouts(
        force,
        10.0,
        minimum_bout_s=4.0,
        merge_gap_s=0.0,
        smooth_s=0.1,
    )

    assert len(bouts) == 2
    assert [round((end - start) / 10.0, 1) for start, end in bouts] == [5.0, 5.0]
    assert -1.0 < threshold < 10.0


def test_http_range_reader_retries_a_temporary_figshare_403(monkeypatch):
    import experiments.reference_fatigue.figshare_task_level as module

    attempts = []

    class Response:
        status = 206
        headers = {"Content-Range": "bytes 0-3/4"}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self, size=-1):
            return b"abcd"

    def fake_urlopen(request, timeout):
        attempts.append(request)
        if len(attempts) == 1:
            raise HTTPError(request.full_url, 403, "temporary signed URL failure", {}, io.BytesIO())
        return Response()

    monkeypatch.setattr(module, "urlopen", fake_urlopen)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    remote = module.HttpRangeFile("https://example.org/archive.zip", 4, block_size=4)

    assert remote.read() == b"abcd"
    assert len(attempts) == 2
    assert remote.range_requests == 1
