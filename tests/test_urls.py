import itertools

import requests

from vit_keras import vit


def test_urls():
    """Ensure that all the weights URLs work."""
    for weights, size in itertools.product(vit.WEIGHTS, vit.SIZES):  # type: ignore
        with requests.get(
            f"{vit.BASE_URL}/ViT-{size}_{weights}.npz", stream=True  # type: ignore
        ) as r:
            assert r.status_code == 200
