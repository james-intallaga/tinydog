import mlx.core as mx

from mini_arithmetic.model import ArithmeticLM, ModelConfig


def test_tiny_model_forward() -> None:
    config = ModelConfig(
        vocab_size=264,
        context_length=16,
        num_layers=2,
        model_dim=32,
        num_heads=4,
        ffn_dim=64,
    )
    model = ArithmeticLM(config)
    output = model(mx.zeros((2, 8), dtype=mx.int32))
    mx.eval(output)
    assert output.shape == (2, 8, 264)


def test_100m_parameter_count() -> None:
    model = ArithmeticLM(ModelConfig())
    assert model.parameter_count() == 99_708_672


def test_300m_parameter_count() -> None:
    config = ModelConfig.from_json("configs/300m.json")
    model = ArithmeticLM(config)
    assert model.parameter_count() == 301_325_312
