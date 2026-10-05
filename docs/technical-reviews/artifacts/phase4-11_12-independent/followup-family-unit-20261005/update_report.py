from pathlib import Path
import datetime
import hashlib
import json
import re

ART = Path(__file__).resolve().parent
ROOT = ART.parents[4]
sha = lambda raw: hashlib.sha256(raw).hexdigest()
target = ROOT / 'docs/technical-reviews/11.12.json'
old = target.read_bytes()
assert sha(old) == '2d01eecb347398175f51397a1b5bf643b144c632a48557970c0d4f7bea64c5d3'
assert old == (ART/'11.12.initial-pass.opaque.json').read_bytes()
report = json.loads(old)  # Only my own report, explicitly allowed for this follow-up.
receipt = json.loads((ART/'issue-reread-receipt.json').read_bytes())
data = json.loads((ART/'split-unit-verification.json').read_bytes())
execution = json.loads((ART/'split-unit-execution.json').read_bytes())
assert execution['exit_code'] == 0
assert report['source_sha256'] == data['source_sha256'] == receipt['source_sha256']
ids = {}
for p in sorted(ART.iterdir()):
    if not p.is_file():
        continue
    identifier = 'family_followup_' + re.sub(r'[^A-Za-z0-9_]', '_', p.name)
    ids[p.name] = identifier
    kind = 'code' if p.suffix == '.py' else 'source_snapshot'
    artifact = {'id':identifier,'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p.read_bytes()),'kind':kind,
                'description':'Actual family/split-unit follow-up evidence: '+p.name+'. Prior PASS is opaque history, not substantive proof.'}
    if p.name == 'split-unit-execution.json':
        artifact.update(kind='execution',command='.venv/bin/python '+(ART/'check_split_units.py').relative_to(ROOT).as_posix(),
                        result='Exit0; original SHA/split hashes match; full-string groups80/10/10 with three offsets each; train character classes0-9, test novel character classes none; both hypothetical two-source whole-family holdouts remove one entire class.',
                        environment=data['environment'])
    report['artifacts'].append(artifact)
report['sources'].append({'id':'family_unit_followup_run','kind':'execution','title':'Original raw split-unit and character-class follow-up recalculation',
                          'verified':True,'artifact_id':ids['split-unit-execution.json']})
family = next(c for c in report['claims'] if c['id']=='family_holdout')
family['status'] = 'needs_clarification'
family['statement'] = '本節要求同一原始字形及其變體整家族切分，以免把練習變體當全新材料；但未明確區分兩圖示範的字元家族與0-99實驗的完整字串家族。'
family['scope'] = 'The conditional label invariance and general dependent-group holdout principle remain verified. Application to the only two original per-class glyphs is insufficiently bounded: whole-family holdout removes an entire class. Actual 0-99 holdout groups whole digit strings, keeps all ten character classes in training and reuses the fixed font; these are different novelty questions that the current text should identify explicitly.'
family['evidence'].append({'source_id':'family_unit_followup_run','locator':'split-unit-verification.json#/split_units, #/actual_test_novelty and #/hypothetical_two_fixture_family_holdout',
                           'supports':'Proves the actual full-string family unit and zero unseen test digit classes; derives the class-holdout consequence for the two sole original glyph families. Does not establish that a model can/cannot predict a novel class.'})
family['artifact_ids'] += [ids['split-unit-execution.json'],ids['split-unit-verification.json'],ids['issue-reread-receipt.json']]
historical = next(c for c in report['claims'] if c['id']=='historical_fixed_font_result')
historical['scope'] += ' Follow-up raw-record verification additionally confirms all 0-9 occur in training and zero test character classes are novel; the score is a full-string generation score on unseen target strings using familiar character classes.'
historical['evidence'].append({'source_id':'family_unit_followup_run','locator':'split-unit-verification.json#/actual_test_novelty','supports':'Only the complete strings are unseen, while digit character classes and font are familiar.'})
historical['artifact_ids'] += [ids['split-unit-execution.json'],ids['split-unit-verification.json']]
report['claims'].append({
    'id':'historical_split_unit_and_class_coverage','kind':'empirical','status':'verified',
    'statement':'The historical OCR split groups entire digit strings and their three offsets; every test digit character class also occurs in training, and the font remains fixed.',
    'location':'course/chapters/11.md:439; raw /results/data/splits/*/records and run_ocr 833-851',
    'scope':'Dataset composition and test novelty, not newly measured OCR accuracy. Unseen full output strings in a character/byte generator do not mean unseen digit-character classes. Holding out original images, holding out a font and holding out complete strings ask separate questions.',
    'evidence':[{'source_id':'ocr_code','locator':'run_ocr 833-851; _manifest 41-53','supports':'Whole full digit text is the family key; offsets stay together and overlapping string groups are rejected.'},
                {'source_id':'draw_digits','locator':'draw_digits 25-36','supports':'Same global glyph lookup for all characters and splits; no font variation axis.'},
                {'source_id':'family_unit_followup_run','locator':'split-unit-verification.json#/split_units and #/actual_test_novelty','supports':'Exact raw-record/string-group hashes checked; ten training digit-character classes, nine test digit classes, no test character class absent in train.'}],
    'artifact_ids':[ids['split-unit-execution.json'],ids['split-unit-verification.json'],ids['section-rechecked.md']],
    'verification':{'method':'executed','expected':'Full-digit-string family equality, disjoint full strings, offsets together, fixed font and holdout character-set coverage correctly distinguished.',
                    'observed':'Train80 strings/240 questions contains0-9; test10 strings/30 questions contains9 familiar digit classes and0 unseen character classes; font lookup unchanged.',
                    'details':'Read only named raw split records/count/sha pointers and code provenance, verified their hashes, recomputed family and per-character sets, and reread AST-located grouping/drawing calculations. No model loaded, trained or re-evaluated.',
                    'denominators':{'train_questions':'240','train_string_families':'80','train_character_classes':'10','validation_questions':'30','validation_string_families':'10','test_questions':'30','test_string_families':'10','test_character_classes':'9','offsets_per_string':'3','unseen_test_character_classes':'0'}}
})
report['verdict'] = 'revise'
report['reviewed_at'] = datetime.datetime.now(datetime.UTC).isoformat()
report['issues'] = [receipt['issue']]
report['followup_record'] = {'question':receipt['question'],'receipt_artifact_id':ids['issue-reread-receipt.json'],
                             'prior_pass_report_sha256':receipt['prior_pass_sha256'],'prior_pass_history_artifact_id':ids['11.12.initial-pass.opaque.json'],
                             'actual_reread':receipt['actual_reread'],'current_source_unchanged':True,'unrelated_verification_repeated':False}
report['checks']['factual_accuracy'] = {'status':'revise','details':'One material scope issue is unresolved: the two glyph-source families would also be class families, whereas the measured experiment holds out whole strings using known characters. General grouping, original matrices, OCR scoring and saved numeric result remain verified.',
                                       'claim_ids':[c['id'] for c in report['claims']]}
report['checks']['limitations'] = {'status':'revise','details':'Current text should explicitly distinguish unseen original-image/glyph sources, unseen classes, unseen fonts and unseen complete strings. The saved2/30 tests only full strings in a fixed font using digit classes already represented in training. No new model inference was needed for this follow-up.',
                                  'claim_ids':['family_holdout','historical_fixed_font_result','historical_split_unit_and_class_coverage']}
report['checks']['numeric_verification']['claim_ids'].append('historical_split_unit_and_class_coverage')
report['checks']['numeric_verification']['details'] += ' Additional stdlib-only follow-up verified train0-9 class coverage and zero unseen test character classes; original numerical evidence unchanged.'
report['checks']['source_verification']['claim_ids'].append('historical_split_unit_and_class_coverage')
report['checks']['source_verification']['details'] += ' Follow-up reread exact same-SHA grouping/drawing code and original official grouping principle; every raw pointer is recorded separately.'
report['verification_boundary'] = receipt['boundary']
target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'report':target.relative_to(ROOT).as_posix(),'verdict':report['verdict'],'source_sha256':report['source_sha256'],
                  'report_sha256':sha(target.read_bytes()),'issues':[i['id'] for i in report['issues']],'claims':len(report['claims'])},ensure_ascii=False,indent=2))
