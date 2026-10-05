messages = [message.copy() for message in toy_conversations()[0]]
messages[1]["content"] = "3"
exercise_x, exercise_y = render_chat(messages)
print("練習有效目標", exercise_y[exercise_y != -100].tolist())
