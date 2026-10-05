from tiny_perceptron.model import TinyLM, ModelConfig

for width in [8, 16]:
    parameters = TinyLM(ModelConfig(width=width)).description()["parameters"]
    for records in [16, 64]:
        batch_size, answers_per_record, steps = 4, 8, 100
        if width == 8 and records == 16:
            answers_per_record, steps = 4, 100
        print(
            {
                "width": width,
                "不同記錄": records,
                "參數格數": parameters,
                "更新步數": steps,
                "有效答案預算": batch_size * answers_per_record * steps,
                "品質": "尚待訓練與量測",
            }
        )
