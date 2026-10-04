from tiny_perceptron.capstone import CapstoneModel, default_config

for dense in (False, True):
    model = CapstoneModel(default_config(dense=dense))
    report = model.description()
    total = report["parameters"]
    print("Dense" if dense else "MoE", "全部參數", total)
    print("邏輯active參數", report["logical_active_parameters"], "FP32權重bytes", total * 4)
