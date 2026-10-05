article_count, checked_chat_count = 1000, 50
article_cost, checked_chat_cost = 1, 5
print("文章整理成本", article_count * article_cost)
print("問答整理成本", checked_chat_count * checked_chat_cost)
print("若1000筆都逐筆核對問答", article_count * checked_chat_cost)
