for name,count in [("MoE",5_447_107),("Dense",2_288_067)]:
 weight_bytes=count*4
 print(name,"FP32權重bytes",weight_bytes,"約MiB",round(weight_bytes/1024**2,2))
 assert weight_bytes=={"MoE":21788428,"Dense":9152268}[name]
print("payload_only_excludes_tied_serialization_duplicates_metadata_KV_activations_workspaces")
