import unittest

import torch

from model import GPT, GPTConfig


class ModelTests(unittest.TestCase):
    def test_forward_and_backward(self) -> None:
        model = GPT(GPTConfig(vocab_size=32, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0))
        inputs = torch.randint(0, 32, (3, 16))
        logits, loss = model(inputs, inputs)
        self.assertEqual(logits.shape, (3, 16, 32))
        self.assertIsNotNone(loss)
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_context_limit_is_enforced(self) -> None:
        model = GPT(GPTConfig(vocab_size=16, block_size=4, n_layer=1, n_head=1, n_embd=8, dropout=0.0))
        with self.assertRaises(ValueError):
            model(torch.zeros((1, 5), dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
