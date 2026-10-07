from tiny_perceptron.data import render_chat
pair = {"prompt": "1+1=?", "chosen": "2", "rejected": "3"}
for key in ["chosen", "rejected"]:
    x, y = render_chat([{"role": "user", "content": pair["prompt"]}, {"role": "assistant", "content": pair[key]}])
    print(key, "輸入", x.tolist(), "目標", y.tolist())
