import requests

from vit_keras import vit


def test_urls_exist():
    for pretrained_top in [True, False]:
        for url, _ in vit.WEIGHTS[pretrained_top].values():  # type: ignore
            r = requests.get(url, stream=True)
            assert r.status_code == 200
