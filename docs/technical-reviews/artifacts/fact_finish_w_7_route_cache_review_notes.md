# W.7 config/ignore source-change recheck

Reviewer: /root/fact_finish_w_7, original owner, fresh review continuation.

Personally reread current W.7 in full; its body remains c201d89e9d6bb1ad26c17a5d74b92f5ab2d17d24a28c92490c668f333e2db229. Personally read complete current zensical.toml and .gitignore and every actual old/new diff. Recovered the old two files from the base Git object only after their full-file SHA exactly matched the prior personally authored report. Preserved the prior report bytes and both full old/new source bytes. No prior execution artifact was rewritten.

The configuration adds 14 navigation targets: 11.14-11.16, 12.13-12.14, chapter-20 and 20.1-20.8. No navigation target was removed; no non-navigation project field changed. docs_dir remains outputs/zensical/docs and site_dir remains outputs/site. Personally read export_course's configuration parsing, output path constraints, generated target mapping, navigation check and CLI. The read-only replay program parsed the real TOML, compared its entire navigation set to the actual lesson-index and DOCUMENTS/chapter generation targets, and checked the new source/Notebook files exist. No website build was run.

The sole ignore-rule delta is .venv-natural/. Existing /data/, /checkpoints/, /outputs/, .cache/ and weight-extension ignores remain. Personally ran git check-ignore -v on representative natural-photo/OCR/speech data paths, checkpoint/output paths, local cache, and .venv-natural. All matched the intended current rules. Also personally read the natural-assistant CLI path options, local asset resolution and cache_dir forwarding; --data-root, --output and --cache-dir are caller-selected. Ignoring a listed root does not mean every arbitrary cache/output directory is ignored.

The original W.7 claims c5/c6/c7 about conventional directory purposes still hold. Added current-source execution evidence to those claims and explicitly updated both source inspection notes/version statements. This is a source and behavior recheck, not a hash-only refresh. Original numeric-index issue/history remains intact. No substantive new performance/quality claim is introduced.

Only own report and fact_finish_w_7 artifacts were written. No teaching content, environment, model/data download, GPU job, training, Git index, other review conclusions or test outputs were touched/read. CLI --help was executed before runner creation and therefore did not create an output directory or load a model.
