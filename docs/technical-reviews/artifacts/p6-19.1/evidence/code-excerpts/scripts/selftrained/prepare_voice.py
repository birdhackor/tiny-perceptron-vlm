scripts/selftrained/prepare_voice.py:L30-L55
30: 
31: REVISION = "40ce77cb32a384e4d50a568e1ec39ac804019d33"
32: PARQUET_SHA256 = "cea0246e1a54afe5a3ab9d9542baa53c7cb45fc8f4e28e7872ebafefb25ba99d"
33: PARQUET_SIZE = 32357378
34: SOURCE_ROOT = f"https://huggingface.co/datasets/PolyAI/minds14/resolve/{REVISION}"
35: INTENTS = (
36:     "abroad",
37:     "address",
38:     "app_error",
39:     "atm_limit",
40:     "balance",
41:     "business_loan",
42:     "card_issues",
43:     "cash_deposit",
44:     "direct_debit",
45:     "freeze",
46:     "high_value_payment",
47:     "joint_account",
48:     "latest_transactions",
49:     "pay_bill",
50: )
51: SELECTED = {"address": 0, "app_error": 1, "card_issues": 2}
52: SPLIT_SEED = "phase5-voice-source-groups-20261006-v1"
53: SYSTEM = "你是教學用的有限客服助手，用繁體中文簡短回答，遵守對話中的格式要求。"
54: ANSWERS = {
55:     "address": {
