families = {
    "train": ["盒子沒有數量資訊，請問有幾球？"],
    "test": ["盒子還沒打開，你能確定裡面有幾顆嗎？", "只看外觀能知道盒內球數嗎"],
}
expected = "資訊不足，請提供數量或可看清的圖片。"
for split, prompts in families.items():
    for prompt in prompts:
        print(split, prompt, "→", expected)
print("訓練問句與測試問句相同", families["train"][0] == families["test"][0])
