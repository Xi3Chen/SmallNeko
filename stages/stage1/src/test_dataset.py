import tempfile
import unittest
from array import array
from pathlib import Path

import torch

from dataset import TokenDataset


class DatasetTests(unittest.TestCase):
    def test_batch_has_next_token_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.bin"
            path.write_bytes(array("H", range(20)).tobytes())
            dataset = TokenDataset(path)
            inputs, targets = dataset.get_batch(4, 5, "cpu", generator=torch.Generator().manual_seed(1))
            self.assertEqual(inputs.shape, (4, 5))
            self.assertTrue(torch.equal(targets[:, :-1], inputs[:, 1:]))


if __name__ == "__main__":
    unittest.main()
