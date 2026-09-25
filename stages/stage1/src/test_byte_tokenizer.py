import json
import tempfile
import unittest
from pathlib import Path

from byte_tokenizer import ByteTokenizer, repair_surrogates


class ByteTokenizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = ByteTokenizer()

    def test_vocab_is_fixed_and_covers_all_bytes(self) -> None:
        self.assertEqual(self.tokenizer.vocab_size, 262)
        self.assertEqual(self.tokenizer.encode("\x00"), [self.tokenizer.byte_offset])
        self.assertEqual(self.tokenizer.encode("🙂"), [i + self.tokenizer.byte_offset for i in "🙂".encode()])

    def test_special_tokens_are_atomic(self) -> None:
        encoded = self.tokenizer.encode("<|im_start|>user")
        self.assertEqual(encoded[0], self.tokenizer.stoi["<|im_start|>"])
        self.assertEqual(self.tokenizer.decode(encoded), "<|im_start|>user")

    def test_unicode_round_trip(self) -> None:
        text = "中文、English、🙂\n<|im_start|>assistant\n测试<|im_end|>"
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)

    def test_skip_special_tokens(self) -> None:
        encoded = self.tokenizer.encode("<|im_start|>你好<|im_end|>")
        self.assertEqual(self.tokenizer.decode(encoded, skip_special_tokens=True), "你好")

    def test_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            self.tokenizer.save(path)
            loaded = ByteTokenizer.load(path)
            self.assertEqual(loaded.tokens, self.tokenizer.tokens)
            self.assertEqual(loaded.encode("你好🙂"), self.tokenizer.encode("你好🙂"))
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["type"], "utf8_byte")

    def test_surrogate_input_is_repaired(self) -> None:
        self.assertEqual(repair_surrogates("阶段1\ud83d\ude42"), "阶段1🙂")
        self.assertEqual(repair_surrogates("阶段1\ud800"), "阶段1�")
        self.assertEqual(
            self.tokenizer.decode(self.tokenizer.encode("阶段1\ud83d\ude42")),
            "阶段1🙂",
        )

    def test_decode_can_drop_partial_utf8_tail(self) -> None:
        encoded = self.tokenizer.encode("中文")
        partial = encoded[:-1]
        self.assertEqual(self.tokenizer.decode(partial), "中�")
        self.assertEqual(self.tokenizer.decode(partial, drop_incomplete_tail=True), "中")


if __name__ == "__main__":
    unittest.main()
