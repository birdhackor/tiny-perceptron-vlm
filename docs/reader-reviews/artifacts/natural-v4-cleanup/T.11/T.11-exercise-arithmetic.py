baseline_bytes = 80000
changed_bytes = 60000
baseline_correct, changed_correct, questions = 4, 3, 5
size_saved = baseline_bytes - changed_bytes
rate_difference = changed_correct / questions - baseline_correct / questions
size_pass = changed_bytes < baseline_bytes
quality_pass = changed_correct >= baseline_correct
print("saved_bytes", size_saved)
print("rate_difference", rate_difference)
print("size_pass", size_pass, "quality_pass", quality_pass, "overall", size_pass and quality_pass)
