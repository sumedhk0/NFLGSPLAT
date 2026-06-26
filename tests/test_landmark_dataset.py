import json
import numpy as np


def _make_dataset(tmp_path):
    import cv2
    from nfl_gsplat.calibration.field_landmarks import NFL_LANDMARKS
    frames_dir = tmp_path / "frames"; frames_dir.mkdir()
    names = [n for n in sorted(NFL_LANDMARKS) if -20 <= NFL_LANDMARKS[n][0] <= 20][:3]
    frames = []
    for fi in range(2):
        img = np.full((1080, 1920, 3), 60, np.uint8)
        cv2.imwrite(str(frames_dir / f"f{fi}.png"), img)
        pts = [{"name": names[0], "uv": [960.0, 540.0]},
               {"name": names[1], "uv": [400.0, 300.0]}]
        frames.append({"file": f"f{fi}.png", "points": pts})
    label = {"image_size": [1920, 1080], "frames": frames}
    p = tmp_path / "labels.json"; p.write_text(json.dumps(label))
    return p, frames_dir, names


def test_dataset_shapes_and_visibility(tmp_path):
    from nfl_gsplat.landmarks.dataset import LandmarkDataset
    from nfl_gsplat.landmarks.schema import LandmarkSchema
    label, frames_dir, names = _make_dataset(tmp_path)
    s = LandmarkSchema(yard_min=-20.0, yard_max=20.0)
    ds = LandmarkDataset(label, frames_dir, s, in_hw=(540, 960), heat_stride=4)
    assert len(ds) == 2
    img, heat, vis = ds[0]
    assert img.shape == (3, 540, 960)
    assert heat.shape == (s.num_classes, 540 // 4, 960 // 4)
    assert vis[s.index(names[0])] == 1.0 and heat[s.index(names[0])].max() > 0.9
    assert vis[s.index(names[2])] == 0.0 and float(heat[s.index(names[2])].max()) == 0.0


def test_dataset_scales_uv_to_input_then_heatmap(tmp_path):
    from nfl_gsplat.landmarks.dataset import LandmarkDataset
    from nfl_gsplat.landmarks.schema import LandmarkSchema
    label, frames_dir, names = _make_dataset(tmp_path)
    s = LandmarkSchema(yard_min=-20.0, yard_max=20.0)
    ds = LandmarkDataset(label, frames_dir, s, in_hw=(540, 960), heat_stride=4)
    _, heat, _ = ds[0]
    ch = heat[s.index(names[0])]
    iy, ix = np.unravel_index(int(ch.argmax()), ch.shape)
    assert abs(ix - (960 / 4) / 2) <= 1 and abs(iy - (540 / 4) / 2) <= 1
