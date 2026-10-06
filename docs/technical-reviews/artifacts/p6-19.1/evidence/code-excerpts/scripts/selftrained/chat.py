scripts/selftrained/chat.py:L27-L110
27: def pixel_roi(value):
28:     try:
29:         coordinates = [int(part) for part in value.split(",")]
30:     except ValueError:
31:         raise argparse.ArgumentTypeError("ROI must be x1,y1,x2,y2 integer pixels") from None
32:     if len(coordinates) != 4 or not (0 <= coordinates[0] < coordinates[2] and 0 <= coordinates[1] < coordinates[3]):
33:         raise argparse.ArgumentTypeError("ROI must be nonnegative x1,y1,x2,y2 with positive area")
34:     return coordinates
35: 
36: 
37: def parser():
38:     result = argparse.ArgumentParser(description=__doc__)
39:     result.add_argument("--model-dir", required=True, help="Directory containing the three safe payloads and manifest")
40:     result.add_argument("--messages", required=True, help="JSON list of public messages ending with the current user")
41:     result.add_argument("--asset-dir", required=True, help="Root for relative image/audio asset paths")
42:     result.add_argument(
43:         "--task", choices=TASKS, default="text", help="Preprocessing/task metadata, never an answer lookup"
44:     )
45:     result.add_argument("--image", help="Public relative image path inside asset-dir")
46:     result.add_argument("--audio", help="Public relative audio path inside asset-dir")
47:     result.add_argument("--roi", type=pixel_roi, help="Public x1,y1,x2,y2 pixel ROI")
48:     result.add_argument("--image-layout", help="JSON file containing public slot geometry")
49:     result.add_argument(
50:         "--modality-message-index", type=int, help="Original user-message index for a new multi-turn asset"
51:     )
52:     result.add_argument("--max-new-tokens", type=int, default=128)
53:     result.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
54:     result.add_argument("--threads", type=int, default=2)
55:     result.add_argument(
56:         "--tools", action="store_true", help="Enable the bounded calculator and same-core second generation"
57:     )
58:     result.add_argument("--manifest-sha256", help="Optional external SHA-256 pin for the manifest itself")
59:     result.add_argument("--repo", help="Optional public HF owner/name model repository")
60:     result.add_argument("--revision", help="Required with repo: immutable lowercase 40-hex commit")
61:     result.add_argument("--prefix", default="", help="Safe-export directory inside the public HF repository")
62:     result.add_argument("--output", help="Also save the complete JSON result to this file")
63:     result.add_argument("--history-output", help="Save actual continued messages as a JSON list for another turn")
64:     return result
65: 
66: 
67: def main(argv=None):
68:     argument_parser = parser()
69:     args = argument_parser.parse_args(argv)
70:     if args.threads < 1:
71:         argument_parser.error("threads must be positive")
72:     if bool(args.repo) != bool(args.revision):
73:         argument_parser.error("repo and its immutable revision must be provided together")
74:     if args.prefix and not args.repo:
75:         argument_parser.error("prefix is used only with a public repo and revision")
76:     torch.set_num_threads(args.threads)
77:     source = None
78:     if args.repo:
79:         source = fetch_public_export(
80:             args.repo, args.revision, args.model_dir, prefix=args.prefix, manifest_sha256=args.manifest_sha256
81:         )
82:     messages = json.loads(Path(args.messages).read_text(encoding="utf-8"))
83:     layout = json.loads(Path(args.image_layout).read_text(encoding="utf-8")) if args.image_layout else None
84:     assistant = InferenceAssistant(
85:         args.model_dir, args.asset_dir, device=args.device, manifest_sha256=args.manifest_sha256
86:     )
87:     result = assistant.reply(
88:         messages,
89:         task=args.task,
90:         max_new_tokens=args.max_new_tokens,
91:         tools=args.tools,
92:         image=args.image,
93:         audio=args.audio,
94:         roi=args.roi,
95:         image_layout=layout,
96:         modality_message_index=args.modality_message_index,
97:     )
98:     if source is not None:
99:         result["public_source"] = source
100:     content = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
101:     for path, text in (
102:         (args.output, content),
103:         (args.history_output, json.dumps(result["messages"], ensure_ascii=False, indent=2) + "\n"),
104:     ):
105:         if path:
106:             target = Path(path)
107:             target.parent.mkdir(parents=True, exist_ok=True)
108:             target.write_text(text, encoding="utf-8")
109:     print(content, end="")
110:     return result
