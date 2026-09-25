import tempfile
import unittest
from array import array
from pathlib import Path

from byte_tokenizer import ByteTokenizer
from prepare_data import encode_file


class PrepareDataTests(unittest.TestCase):
    def test_special_markers_near_chunk_boundaries_are_preserved(self) -> None:
        tokenizer = ByteTokenizer()
        texts = (
            "a" * (1024 * 1024 - 3) + "<|im_start|>assistant\n答案",
            "a" * (1024 * 1024 - 18) + "<|im_end|>12345678",
        )
        with tempfile.TemporaryDirectory() as directory:
            for index, text in enumerate(texts):
                source = Path(directory) / f"source-{index}.txt"
                output = Path(directory) / f"tokens-{index}.bin"
                source.write_text(text, encoding="utf-8")
                count = encode_file(tokenizer, source, output)
                encoded = array("H")
                encoded.frombytes(output.read_bytes())
                expected = tokenizer.encode(text)
                self.assertEqual(count, len(expected))
                self.assertEqual(encoded.tolist(), expected)


if __name__ == "__main__":
    unittest.main()
