import json
import tempfile
import unittest
from pathlib import Path

from char_tokenizer import CharTokenizer


class CharTokenizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = CharTokenizer.from_text(
            "<|im_start|>user\n你好<|im_end|>\n普通文本"
        )

    def test_special_tokens_are_atomic(self) -> None:
        encoded = self.tokenizer.encode("<|im_start|>user")
        self.assertEqual(encoded[0], self.tokenizer.stoi["<|im_start|>"])
        self.assertEqual(self.tokenizer.decode(encoded), "<|im_start|>user")

    def test_unknown_character_uses_unk(self) -> None:
        encoded = self.tokenizer.encode("不存在的字")
        self.assertIn(self.tokenizer.stoi["<unk>"], encoded)

    def test_bos_and_eos(self) -> None:
        encoded = self.tokenizer.encode("你好", add_bos=True, add_eos=True)
        self.assertEqual(encoded[0], self.tokenizer.stoi["<bos>"])
        self.assertEqual(encoded[-1], self.tokenizer.stoi["<eos>"])

    def test_skip_special_tokens(self) -> None:
        encoded = self.tokenizer.encode("<|im_start|>你好<|im_end|>")
        self.assertEqual(self.tokenizer.decode(encoded, skip_special_tokens=True), "你好")

    def test_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            self.tokenizer.save(path)
            loaded = CharTokenizer.load(path)
            self.assertEqual(loaded.tokens, self.tokenizer.tokens)
            self.assertEqual(loaded.decode(loaded.encode("你好")), "你好")
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["type"], "unicode_character")


if __name__ == "__main__":
    unittest.main()
