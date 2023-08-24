from os import remove

import pytest
import torch as t
from torchvision.models import resnet

from hooks import FeatureExtractor, LayerDiffRMS, remove_hooks

model = resnet.resnet50()
x = t.randn(size=(1, 3, 224, 224))


def test_feature_extractor(model=model, x=x):
    model = FeatureExtractor(model, "conv1")
    out = model(x)
    assert model._features.shape == (1, 64, 112, 112)

    remove_hooks(model)

    assert len(model._forward_hooks) == 0


def test_layer_diff_rms(model=model, x=x):
    model = LayerDiffRMS(model, "bn1")
    out = model(x)
    assert model._rms is not None

    remove_hooks(model)

    assert len(model._forward_hooks) == 0


def test_layer_diff_rms_exception(model=model, x=x):
    model = LayerDiffRMS(model, "conv1")
    with pytest.raises(ValueError):
        out = model(x)
