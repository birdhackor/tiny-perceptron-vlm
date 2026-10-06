scripts/selftrained/train.py:L235-L281
235: def export_inference(output_dir, *, required=False):
236:     """公開檔只含選定模型與契約，不帶 optimizer/RNG 或訓練標籤。"""
237:     selected = output_dir / "best.pt"
238:     if not selected.exists():
239:         return False
240:     try:
241:         from safetensors.torch import save_file
242:     except ImportError:
243:         if required:
244:             raise RuntimeError("--export-inference 需要 safetensors 純權重格式套件") from None
245:         return False
246:     from tiny_perceptron.selftrained.dataset import file_sha256
247:     from tiny_perceptron.selftrained.tokenizer import CharacterTokenizer
248: 
249:     checkpoint = torch.load(selected, map_location="cpu", weights_only=False)
250:     # tied embeddings 的兩個名稱保留；clone 避免 safe writer 的共享 storage 歧義。
251:     tensors = {name: value.detach().cpu().contiguous().clone() for name, value in checkpoint["model"].items()}
252:     save_file(
253:         tensors,
254:         str(output_dir / "model.safetensors"),
255:         metadata={"origin": "all-neural-weights-random", "schema": CHECKPOINT_SCHEMA},
256:     )
257:     (output_dir / "model-config.json").write_text(json.dumps(checkpoint["config"], indent=2) + "\n", encoding="utf-8")
258:     CharacterTokenizer.from_dict(checkpoint["tokenizer"]).save(output_dir / "tokenizer.json")
259:     manifest = {
260:         "schema": CHECKPOINT_SCHEMA,
261:         "selected_checkpoint_sha256": file_sha256(selected),
262:         "files": {
263:             name: file_sha256(output_dir / name)
264:             for name in ("model.safetensors", "model-config.json", "tokenizer.json")
265:         },
266:         "stage": checkpoint["stage"],
267:         "selected_step": checkpoint["step"],
268:         "origin": checkpoint["origin"],
269:         "preprocess_version": checkpoint["preprocess_version"],
270:         "data_sha256": checkpoint["data_sha256"],
271:         "asset_sha256": checkpoint["asset_sha256"],
272:         "tokenizer_sha256": checkpoint["tokenizer_sha256"],
273:         "freeze_perception_backbones": checkpoint.get("training_options", {}).get("freeze_perception_backbones", False),
274:         "sampling_mode": checkpoint.get("training_options", {}).get("sampling_mode", "bucket"),
275:         **checkpoint_language_objective(checkpoint),
276:         "selection": "validation_loss",
277:     }
278:     (output_dir / "inference-manifest.json").write_text(
279:         json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
280:     )
281:     return True

scripts/selftrained/train.py:L689-L811
689: def main(argv=None):
690:     args = parser().parse_args(argv)
691:     native_weight = args.native_voice_loss_weight
692:     weighted = validate_loss_weights(args.stage, args.tool_loss_weight, args.numeric_run_loss_weight, native_weight)
693:     if native_weight == 1:
694:         # Preserve all old serialized args/options keys, explicit native1 included.
695:         del args.native_voice_loss_weight
696:     elif (
697:         (args.tool_loss_weight, args.numeric_run_loss_weight, native_weight) != (4, 1, 4)
698:         or args.steps > 4000
699:         or args.batch_size != 16
700:         or args.context != 512
701:         or args.learning_rate != 0.0002
702:         or not args.freeze_perception_backbones
703:         or args.sampling_mode != "task-family"
704:         or args.max_tokens is not None
705:     ):
706:         raise ValueError(
707:             "native voice candidate requires fixedtool4/numeric1/native4, <=4000 steps, batch16/context512/LR0.0002 and frozen task-family"
708:         )
709:     objective_metadata = language_objective_metadata(vars(args))
710:     args.language_objective_policy = objective_metadata["language_objective_policy"]
711:     if args.resume and args.init_checkpoint:
712:         raise ValueError("resume 與 init-checkpoint 不能同時使用")
713:     if weighted and not (args.resume or args.init_checkpoint):
714:         raise ValueError("nondefault joint 必須 init own selected joint checkpoint 或 exact same-objective resume")
715:     if args.stage != "joint" and (args.freeze_perception_backbones or args.sampling_mode != "bucket"):
716:         raise ValueError("freeze-perception-backbones / task-family sampling 只可用於 joint")
717:     if min(args.steps, args.batch_size, args.context, args.eval_every, args.save_every, args.threads) < 1:
718:         raise ValueError("步數、batch、context、interval、threads 必須為正")
719:     torch.set_num_threads(args.threads)
720:     seed_all(args.seed)
721:     from tiny_perceptron.selftrained.model import LimitedAssistant, SelftrainedConfig
722:     from tiny_perceptron.selftrained.tokenizer import CharacterTokenizer
723: 
724:     all_records = read_records(args.records)
725:     hashes = data_fingerprints(args.records)
726:     asset_hashes = asset_fingerprints(all_records, args.asset_dir)
727:     records = stage_records(all_records, args.stage, "train")
728:     validation = stage_records(all_records, args.stage, "validation")
729:     if not records or not validation:
730:         raise ValueError(f"{args.stage} 需要 train 與 validation，不能拿 test 填補")
731:     source_path = args.resume or args.init_checkpoint
732:     source = torch.load(source_path, map_location="cpu", weights_only=False) if source_path else None
733:     if source is not None and source.get("schema") != CHECKPOINT_SCHEMA:
734:         raise ValueError("只接受本從零管線 checkpoint")
735:     if source is not None and source.get("preprocess_version") != PREPROCESS_VERSION:
736:         raise ValueError("checkpoint preprocess_version 與目前模態前處理不同")
737:     tokenizer = CharacterTokenizer.from_dict(source["tokenizer"]) if source else train_tokenizer(all_records)
738:     if source is not None and tokenizer_sha(tokenizer) != source["tokenizer_sha256"]:
739:         raise ValueError("checkpoint tokenizer 指紋不匹配")
740:     if source and source["data_sha256"] != hashes:
741:         raise ValueError("資料指紋不同；請建立新實驗而非沿用 frozen stage")
742:     if source and source["asset_sha256"] != asset_hashes:
743:         raise ValueError("模態資產指紋不同")
744:     if weighted:
745:         validate_weighted_source(source, args)
746:     native_source_binding = None
747:     if native_weight != 1 and args.init_checkpoint:
748:         native_source_binding = validate_native_source(source_path, source, args)
749:     if args.resume:
750:         validate_resume_language_objective(source, vars(args))
751:         if native_weight != 1:
752:             for name in ("eval_every", "save_every"):
753:                 if source["training_options"].get(name, 100) != getattr(args, name):
754:                     raise ValueError(f"native exact resume 改變 {name}")
755:     config_values = (
756:         dict(source["config"]) if source else (json.loads(Path(args.config).read_text()) if args.config else {})
757:     )
758:     if source and config_values["architecture"] != args.architecture:
759:         raise ValueError("Dense/MoE 不能混用權重")
760:     config_values.update(vocab_size=tokenizer.vocab_size, architecture=args.architecture, max_length=args.context)
761:     config = SelftrainedConfig(**config_values)
762:     model = LimitedAssistant(config).to(args.device)
763:     if source:
764:         model.load_state_dict(source["model"])
765:     parameters = set_trainable(model, args.stage, args.freeze_perception_backbones)
766:     optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
767:     encoder = RecordEncoder(tokenizer, args.asset_dir, args.context, args.device)
768:     # 先審查所有 stage train/val 長度；不讀 test 模態或 target。
769:     for record in records + validation:
770:         row = encoder.encode(record, pretrain=args.stage == "pretrain")
771:         if tokenizer.unk_id in row["input_ids"]:
772:             # validation 可有未知字，train 不得因 stage 增詞而靜默退化。
773:             if record["split"] == "train":
774:                 raise ValueError(f"train 有 tokenizer 未見字: {record['id']}")
775:     sampler = BalancedSampler(records, args.seed, mode=args.sampling_mode)
776:     step, token_count, target_token_count, best_val = 0, 0, 0, math.inf
777:     history = list(source.get("stage_history", [])) if source else []
778:     origin = source.get("origin") if source else {"kind": "all-neural-weights-random", "seed": args.seed}
779:     if args.resume:
780:         if source["stage"] != args.stage or source["config"] != dataclasses.asdict(config):
781:             raise ValueError("resume 必須使用相同 stage/config")
782:         old = source["training_options"]
783:         for name in ("batch_size", "seed", "learning_rate", "weight_decay", "perception_weight", "router_weight"):
784:             if old[name] != getattr(args, name):
785:                 raise ValueError(f"resume 改變 {name}；請使用 init-checkpoint 明確開始新階段")
786:         for name, default in (("freeze_perception_backbones", False), ("sampling_mode", "bucket")):
787:             if old.get(name, default) != getattr(args, name):
788:                 raise ValueError(f"resume 改變 {name}；請使用 init-checkpoint 明確開始新階段")
789:         optimizer.load_state_dict(source["optimizer"])
790:         sampler.load_state_dict(source["sampler"])
791:         restore_rng(source["rng"])
792:         step, token_count, target_token_count = source["step"], source["tokens"], source["target_tokens"]
793:         best_val = source["best_validation_loss"]
794:     elif source:
795:         history.append(
796:             {
797:                 "stage": source["stage"],
798:                 "step": source["step"],
799:                 "tokens": source["tokens"],
800:                 "checkpoint": str(source_path),
801:                 "checkpoint_sha256": file_sha256(source_path),
802:                 "selection": "validation_loss"
803:                 if source.get("checkpoint_kind") == "selected_validation_best"
804:                 else "explicit_init_checkpoint",
805:             }
806:         )
807:         if weighted:
808:             initialization = {
809:                 "source_stage": source["stage"],
810:                 "loaded_step": source["step"],
811:                 "source_checkpoint_sha256": file_sha256(source_path),
