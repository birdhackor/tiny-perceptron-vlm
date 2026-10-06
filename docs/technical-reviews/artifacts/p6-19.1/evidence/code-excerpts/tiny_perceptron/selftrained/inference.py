tiny_perceptron/selftrained/inference.py:L74-L173
74: def verify_export(model_dir, *, manifest_sha256=None):
75:     """Verify every payload before parsing model configuration or allocating weights."""
76:     root = Path(model_dir)
77:     manifest_path = root / MANIFEST_FILE
78:     actual_manifest_hash = file_sha256(manifest_path)
79:     if manifest_sha256 is not None and actual_manifest_hash != _sha256(manifest_sha256):
80:         raise ValueError("Inference manifest SHA-256 mismatch")
81:     manifest = _read_json(manifest_path)
82:     if not isinstance(manifest, dict) or manifest.get("schema") != EXPORT_SCHEMA:
83:         raise ValueError("Unknown inference export schema")
84:     files = manifest.get("files")
85:     if not isinstance(files, dict) or set(files) != set(PAYLOAD_FILES):
86:         raise ValueError("Manifest must bind exactly the three safe inference payloads")
87:     if manifest.get("origin", {}).get("kind") != "all-neural-weights-random":
88:         raise ValueError("Export must declare the from-scratch neural origin")
89:     if manifest.get("preprocess_version") != PREPROCESS_VERSION:
90:         raise ValueError("Export preprocessing version differs from this encoder")
91:     if manifest.get("selection") != "validation_loss":
92:         raise ValueError("Only a validation-selected export is supported")
93:     _sha256(manifest.get("selected_checkpoint_sha256"))
94:     for name in PAYLOAD_FILES:
95:         expected = _sha256(files[name])
96:         if file_sha256(root / name) != expected:
97:             raise ValueError(f"Inference payload SHA-256 mismatch: {name}")
98:     return manifest, actual_manifest_hash
99: 
100: 
101: def fetch_public_export(repo, revision, model_dir, *, prefix="", manifest_sha256=None):
102:     """Fetch four public files at one immutable commit, with authentication disabled.
103: 
104:     The release repository/revision are supplied by the user; no unpublished
105:     release is guessed. No remote code or pickle checkpoint is downloaded.
106:     """
107:     if not isinstance(revision, str) or REVISION_PATTERN.fullmatch(revision) is None:
108:         raise ValueError("HF revision must be an immutable lowercase 40-hex commit")
109:     if not isinstance(repo, str) or re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo) is None:
110:         raise ValueError("HF repo must be an owner/name model repository")
111:     path_prefix = PurePosixPath(prefix)
112:     if path_prefix.is_absolute() or ".." in path_prefix.parts or "\\" in prefix:
113:         raise ValueError("HF prefix must be a relative repository directory")
114:     from huggingface_hub import hf_hub_download
115: 
116:     downloaded = {}
117:     for name in (MANIFEST_FILE, *PAYLOAD_FILES):
118:         filename = str(path_prefix / name)
119:         downloaded[name] = Path(
120:             hf_hub_download(repo_id=repo, filename=filename, revision=revision, repo_type="model", token=False)
121:         )
122:     # Cached paths keep their repository directory, and form one complete export.
123:     source_dir = downloaded[MANIFEST_FILE].parent
124:     verify_export(source_dir, manifest_sha256=manifest_sha256)
125:     destination = Path(model_dir)
126:     for name, source in downloaded.items():
127:         target = destination / name
128:         if target.exists() and file_sha256(target) != file_sha256(source):
129:             raise FileExistsError(f"Refusing to overwrite a different local inference file: {target}")
130:     destination.mkdir(parents=True, exist_ok=True)
131:     for name, source in downloaded.items():
132:         target = destination / name
133:         if not target.exists():
134:             shutil.copyfile(source, target)
135:     verify_export(destination, manifest_sha256=manifest_sha256)
136:     return {"repo": repo, "revision": revision, "prefix": prefix, "authentication": "disabled"}
137: 
138: 
139: def _public_history(messages, *, require_user=True):
140:     if not isinstance(messages, list) or not messages:
141:         raise ValueError("Messages must be a nonempty JSON list")
142:     history = copy.deepcopy(messages)
143:     for message in history:
144:         if not isinstance(message, dict):
145:             raise ValueError("Each message must be a JSON object")
146:         if set(message) - {"role", "content", *PUBLIC_MODAL_KEYS}:
147:             raise ValueError("Messages accept role/content and public image/audio/ROI/layout metadata only")
148:         if message.get("role") not in ("system", "user", "assistant", "tool"):
149:             raise ValueError("Unknown message role")
150:         if not isinstance(message.get("content"), str):
151:             raise ValueError("Message content must be text")
152:         if PUBLIC_MODAL_KEYS & message.keys() and message["role"] != "user":
153:             raise ValueError("Public modality metadata belongs to its original user message")
154:         for key in ("image", "audio"):
155:             if key in message and (not isinstance(message[key], str) or not message[key]):
156:                 raise ValueError("Modality asset paths must be nonempty relative strings")
157:         roi = message.get("roi")
158:         if roi is not None and not _pixel_box(roi):
159:             raise ValueError("ROI must be nonnegative pixel [x1,y1,x2,y2] with positive area")
160:         if ("roi" in message or "image_layout" in message) and "image" not in message:
161:             raise ValueError("ROI/layout metadata must accompany its image in the same message")
162:         if "image_layout" in message:
163:             layout = message["image_layout"]
164:             if not isinstance(layout, dict) or set(layout) - {"axis", "slots"}:
165:                 raise ValueError("image_layout accepts public axis/slots geometry only")
166:             slots = layout.get("slots")
167:             if not isinstance(slots, list) or len(slots) != 2 or not all(_pixel_box(box) for box in slots):
168:                 raise ValueError("image_layout requires two nonnegative pixel slot boxes")
169:             if layout.get("axis") not in (None, "horizontal", "vertical"):
170:                 raise ValueError("image_layout axis must be horizontal or vertical")
171:     if require_user and history[-1]["role"] != "user":
172:         raise ValueError("Input history must end with the current user, without an assistant target")
173:     return history

tiny_perceptron/selftrained/inference.py:L218-L328
218:     def __init__(self, model_dir, asset_dir, *, device="cpu", manifest_sha256=None):
219:         from safetensors import safe_open
220:         from safetensors.torch import load_file
221: 
222:         if device not in ("cpu", "cuda"):
223:             raise ValueError("Inference device must be cpu or cuda")
224:         if device == "cuda" and not torch.cuda.is_available():
225:             raise RuntimeError("CUDA was requested but is unavailable")
226:         root = Path(model_dir)
227:         manifest, manifest_hash = verify_export(root, manifest_sha256=manifest_sha256)
228:         config = SelftrainedConfig(**_read_json(root / "model-config.json"))
229:         tokenizer = CharacterTokenizer.load(root / "tokenizer.json")
230:         if tokenizer.vocab_size != config.vocab_size:
231:             raise ValueError("Tokenizer vocabulary differs from model configuration")
232:         with safe_open(root / "model.safetensors", framework="pt", device="cpu") as archive:
233:             metadata = archive.metadata() or {}
234:         if metadata.get("origin") != "all-neural-weights-random" or metadata.get("schema") != EXPORT_SCHEMA:
235:             raise ValueError("Safetensors metadata differs from the from-scratch export contract")
236:         state = load_file(root / "model.safetensors", device="cpu")
237:         model = LimitedAssistant(config)
238:         expected = model.state_dict()
239:         if set(state) != set(expected):
240:             raise ValueError("Safetensors keys differ from the complete multimodal model")
241:         for name, tensor in state.items():
242:             if tensor.shape != expected[name].shape or tensor.dtype != expected[name].dtype:
243:                 raise ValueError(f"Safetensors shape/dtype mismatch: {name}")
244:             if not torch.isfinite(tensor).all():
245:                 raise ValueError(f"Safetensors contains nonfinite weights: {name}")
246:         if config.tied and not torch.equal(state["lm.embedding.weight"], state["lm.output.weight"]):
247:             raise ValueError("Tied embedding/output tensors must agree")
248:         verify_export(root, manifest_sha256=manifest_hash)
249:         model.load_state_dict(state, strict=True)
250:         self.model = model.to(device).eval()
251:         self.encoder = RecordEncoder(tokenizer, asset_dir, config.max_length, device)
252:         self.receipt = {
253:             "manifest_sha256": manifest_hash,
254:             "manifest_hash_pinned": manifest_sha256 is not None,
255:             "files": manifest["files"],
256:             "origin": manifest["origin"],
257:             "stage": manifest["stage"],
258:             "selected_step": manifest["selected_step"],
259:             "selected_checkpoint_sha256": manifest["selected_checkpoint_sha256"],
260:             "preprocess_version": manifest["preprocess_version"],
261:             "config": _read_json(root / "model-config.json"),
262:             "parameters": model.description()["parameters"],
263:             "generation_policy": model.description()["generation_policy"],
264:         }
265: 
266:     @torch.no_grad()
267:     def reply(self, messages, *, task="text", max_new_tokens=128, tools=False, **public_metadata):
268:         if task not in TASKS:
269:             raise ValueError("Unknown finite assistant task")
270:         if type(max_new_tokens) is not int or max_new_tokens < 1:
271:             raise ValueError("max_new_tokens must be a positive integer")
272:         history = attach_public_metadata(messages, **public_metadata)
273:         if task == "ocr" and history[-1].get("image") and "roi" not in history[-1]:
274:             raise ValueError("OCR images require a public ROI")
275:         calls = []
276: 
277:         def generate(actual_history):
278:             actual_history = _public_history(actual_history, require_user=False)
279:             # Unlike a dataset record this contains no final assistant target or supervision.
280:             record = {"id": "user-inference", "task": task, "messages": actual_history}
281:             encoded = self.encoder.encode(record, generation=True, messages=actual_history)
282:             if len(encoded["input_ids"]) + max_new_tokens > self.encoder.context:
283:                 raise ValueError("Complete history plus generation exceeds model context; history is not truncated")
284:             ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=self.encoder.device)
285:             generated = self.model.generate(
286:                 ids,
287:                 modalities=[self.encoder.to_device(encoded["modalities"])],
288:                 max_new_tokens=max_new_tokens,
289:                 eos_id=self.encoder.tokenizer.eos_id,
290:             )
291:             tokens = generated[0].detach().cpu().tolist()
292:             output = self.encoder.tokenizer.decode(tokens, skip_special_tokens=True)
293:             report = generation_report(self.encoder.tokenizer, tokens)
294:             calls.append(
295:                 {
296:                     "call_index": len(calls) + 1,
297:                     "model_sha256": self.receipt["files"]["model.safetensors"],
298:                     "prompt_messages": actual_history,
299:                     "prompt_ids": encoded["input_ids"],
300:                     "prompt_unknown_tokens": encoded["input_ids"].count(self.encoder.tokenizer.unk_id),
301:                     "generated_ids": tokens,
302:                     "raw_output": output,
303:                     "eos": report["eos"],
304:                     "invalid_control_tokens": report["invalid_special_tokens"],
305:                     "generation_status": "invalid_special_tokens"
306:                     if report["invalid_special_tokens"]
307:                     else "eos"
308:                     if report["eos"]
309:                     else "token_limit",
310:                     "modality_kinds": [payload["kind"] for payload in encoded["modalities"]],
311:                 }
312:             )
313:             return output
314: 
315:         trace = run_tool_loop(generate, history, tools_enabled=tools)
316:         continued = history.copy()
317:         if trace["tool_message"] is not None:
318:             continued.extend([{"role": "assistant", "content": trace["initial_output"]}, trace["tool_message"]])
319:         continued.append({"role": "assistant", "content": trace["final_output"]})
320:         return {
321:             "schema": INFERENCE_SCHEMA,
322:             "model": self.receipt,
323:             "task": task,
324:             "answer": trace["final_output"],
325:             "messages": continued,
326:             "generations": calls,
327:             "tool_trace": trace,
328:         }
