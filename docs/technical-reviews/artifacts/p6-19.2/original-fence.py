from tiny_perceptron.selftrained.model import LimitedAssistant, SelftrainedConfig

for architecture in ("moe", "dense"):
    config = SelftrainedConfig(
        vocab_size=550, width=256, layers=4, heads=4, kv_heads=2,
        ffn_hidden=512, experts=4, top_k=2, max_length=512,
        architecture=architecture,
    )
    model = LimitedAssistant(config)
    report = model.description()
    print(architecture, "全部", report["parameters"], "文字", report["language_parameters"])
    print("每位置的邏輯啟用文字參數", report["active_language_parameters_per_token"])
    print("嵌入與輸出共用同一張表", model.lm.embedding.weight is model.lm.output.weight)
