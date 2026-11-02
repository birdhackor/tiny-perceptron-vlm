total_parameters = 30_000_000_000
active_parameters = 3_000_000_000
bytes_per_parameter = 2
stored = total_parameters * bytes_per_parameter
active_only = active_parameters * bytes_per_parameter
print("全部權重名義值GB", stored / 10**9)
print("全部權重名義值GiB", round(stored / 1024**3, 2))
print("只按啟用量算GB", active_only / 10**9)
