import unittest

from format_chat import conversation_text, format_records


class FormatChatTests(unittest.TestCase):
    def test_instruction_output_format(self) -> None:
        text = conversation_text({"instruction": "问题", "output": "答案"})
        self.assertIn("<|im_start|>user\n问题", text)
        self.assertIn("<|im_start|>assistant\n答案", text)
        self.assertTrue(text.endswith("<|im_end|>\n"))

    def test_messages_and_sharegpt_formats(self) -> None:
        messages = conversation_text(
            {"messages": [{"role": "user", "content": "问"}, {"role": "assistant", "content": "答"}]}
        )
        sharegpt = conversation_text(
            {"conversations": [{"from": "human", "value": "问"}, {"from": "gpt", "value": "答"}]}
        )
        self.assertEqual(messages, sharegpt)

    def test_invalid_and_duplicate_records_are_skipped(self) -> None:
        records, skipped = format_records(
            [
                {"instruction": "相同问题", "output": "答案一"},
                {"instruction": "相同问题", "output": "答案二"},
                {"instruction": "空回答", "output": ""},
                {"unknown": "字段"},
            ]
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(skipped, 3)


if __name__ == "__main__":
    unittest.main()
