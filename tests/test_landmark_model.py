import pytest


@pytest.mark.slow
def test_landmark_net_forward_shape():
    import torch
    from nfl_gsplat.landmarks.model import LandmarkNet
    net = LandmarkNet(num_classes=7, stride=4)
    x = torch.zeros(2, 3, 540, 960)
    y = net(x)
    assert y.shape == (2, 7, 135, 240)
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0
