"""Final narrow checks for historical paired validation and command documentation."""
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).with_name('additional-inspections-v1.json')

def rec(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def read(p):return json.loads(p.read_text())

def main():
    base=ROOT/'docs/natural-assistant/evidence/v4-runtime/validation-37219466611/blind-review'
    scores=read(base/'scored/scores.json');selection=read(ROOT/'docs/natural-assistant/v4/selection.json')
    variants={}
    for name,v in scores['variants'].items():
        p=base/f'raw/generations-{name}.json';rows=read(p)
        assert len(rows)==132
        variants[name]={'original_generations':rec(p),'rows':len(rows),'ended_with_eos':sum(r['ended_with_eos'] for r in rows),
                        'truncated':sum(r['truncated'] for r in rows),'reported_manual_correct_counts':v['correct_counts'],
                        'recomputed_weighted_numerator':sum(v['correct_counts'][kind]*weight for kind,weight in scores['integer_weights'].items())}
        assert variants[name]['recomputed_weighted_numerator']==v['primary_numerator']
    scratch=Path('/workspace/work/reference/omni-P5-validation/git-command-probe');scratch.mkdir(exist_ok=True)
    commands=[]
    def run(args):
        p=subprocess.run(args,cwd=scratch,text=True,capture_output=True);commands.append({'argv':args,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr});assert p.returncode==0
    run(['git','init','--quiet'])
    (scratch/'probe.txt').write_text('review-only local commit\n')
    run(['git','status','--short']);run(['git','add','probe.txt'])
    run(['git','-c','user.name=Validation Probe','-c','user.email=validation-probe@example.invalid','-c','commit.gpgsign=false','commit','--quiet','-m','Review-only probe'])
    run(['git','status','--porcelain'])
    docs=Path('/workspace/work/reference/omni-P5-validation')
    out={'created_at':datetime.now(timezone.utc).isoformat(),'reviewer_task':'/root/omni_p5_technical_validation',
         'environment':{'python':platform.python_version(),'device':'CPU; no model execution'},
         'natural_validation':{'raw_result':rec(base/'raw/result.json'),'scoring_result':rec(base/'scored/scores.json'),'selection':rec(ROOT/'docs/natural-assistant/v4/selection.json'),
                               'selected_variant':selection['selected_variant'],'variants':variants,
                               'limit':'Weighted totals and EOS were independently recomputed; photo/chat semantic labels are reported original measurement labels and were not all regraded.'},
         'official_command_sources':[{'source':rec(docs/'powershell-automatic-variables.html'),'url':'https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_automatic_variables?view=powershell-7.5','version':'PowerShell7.5; $LASTEXITCODE section; accessed2026-10-30','excerpt':'Contains the exit code of the last native program or PowerShell script that ran.'},
                                     {'source':rec(docs/'git-v2.52.0-user.adoc'),'url':'https://raw.githubusercontent.com/git/git/v2.52.0/Documentation/config/user.adoc','version':'Gitv2.52.0; user.name/user.email; accessed2026-10-30','excerpt':'The user.name and user.email variables determine what ends up in the author and committer fields of commit objects.'}],
         'git_command_probe':commands,'scope':'Only synthetic Git-external repository mutated. No changes to project Git index/config/HEAD or global identity. No model forwards.',
         'acquisition_note':'git-scm.com/book official HTML returned HTTP403; the exact tagged official Git source above was read instead.',
         'reviewer_script_corrections':['An initial local audit compared in-memory integer JSON keys to parsed string keys; corrected by JSON normalization before comparing. The data audit then matched exactly.',
                                        'An initial local pytest-help excerpt selector expected -r without a comma; corrected to the actual help spelling -r, --report-chars. Both synthetic kernel probes had already succeeded.']}
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'output':str(OUT.relative_to(ROOT)),'bytes':OUT.stat().st_size,'sha256':hashlib.sha256(OUT.read_bytes()).hexdigest(),'natural_variants':{k:{x:v[x] for x in ['rows','ended_with_eos','truncated','recomputed_weighted_numerator']} for k,v in variants.items()}},ensure_ascii=False))

if __name__=='__main__':main()
