from mini_arithmetic.tokenizer import ArithmeticTokenizer


def test_round_trip_and_digit_tokens() -> None:
    tokenizer = ArithmeticTokenizer()
    text = "Teacher: 347 + 286? π\nAnswer: 633\n"
    tokens = tokenizer.encode(text, bos=True, eos=True)
    assert tokenizer.decode(tokens) == text
    assert len(tokenizer.encode("347")) == 3
    assert tokenizer.vocab_size == 264


def test_dialogue_roles_are_dedicated_tokens() -> None:
    tokenizer = ArithmeticTokenizer()
    tokens = tokenizer.encode_dialogue(
        [
            ("user", "What is 2 + 2?"),
            ("assistant", "4"),
            ("teacher", "Correct."),
        ]
    )
    assert tokenizer.USER in tokens
    assert tokenizer.ASSISTANT in tokens
    assert tokenizer.TEACHER in tokens
    assert tokenizer.decode(tokens) == "What is 2 + 2?4Correct."
