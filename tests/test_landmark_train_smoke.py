import json
import numpy as np
import pytest


@pytest.mark.slow
def test_train_overfits_tiny(tmp_path):
    import cv2
    from nfl_gsplat.calibration.field_landmarks import NFL_LANDMARKS
    from nfl_gsplat.landmarks.schema import LandmarkSchema
    from nfl_gsplat.landmarks.train import train
    frames_dir = tmp_path / "frames"; frames_dir.mkdir()
    names = [n for n in sorted(NFL_LANDMARKS) if -20 <= NFL_LANDMARKS[n][0] <= 20][:3]
    recs = []
    for fi in range(2):
        cv2.imwrite(str(frames_dir / f"f{fi}.png"), np.full((1080, 1920, 3), 60, np.uint8))
        recs.append({"file": f"f{fi}.png",
                     "points": [{"name": names[0], "uv": [960.0, 540.0]}]})
    label = tmp_path / "labels.json"
    label.write_text(json.dumps({"image_size": [1920, 1080], "frames": recs}))
    s = LandmarkSchema(yard_min=-20.0, yard_max=20.0)
    out = tmp_path / "run"
    ck = train(label, frames_dir, s, out_dir=out, epochs=3, batch_size=2, lr=1e-3,
               device="cpu", resume=False)
    assert ck.exists() and (out / "ckpt_last.pt").exists()
    ck2 = train(label, frames_dir, s, out_dir=out, epochs=4, batch_size=2, lr=1e-3,
                device="cpu", resume=True)
    assert ck2.exists()
