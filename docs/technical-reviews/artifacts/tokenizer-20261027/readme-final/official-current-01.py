import urllib.request,hashlib,re,json
from pathlib import Path
from datetime import datetime,UTC
s=Path('/workspace/work/tokenizer-dynamic-20261027/readme-final'); out=Path('docs/technical-reviews/artifacts/tokenizer-20261027/readme-final')
urls={
 'mathjax-hosting':'https://raw.githubusercontent.com/mathjax/MathJax-docs/master/web/hosting.rst',
 'mathjax-typeset':'https://raw.githubusercontent.com/mathjax/MathJax-docs/master/web/typeset.rst',
 'uv-sync':'https://raw.githubusercontent.com/astral-sh/uv/main/docs/concepts/projects/sync.md',
 'git-lfs-skip':'https://raw.githubusercontent.com/git-lfs/git-lfs/main/docs/man/git-lfs-config.adoc',
 'git-windows':'https://raw.githubusercontent.com/git-for-windows/git/main/compat/mingw.c',
 'colab-faq':'https://colab.research.google.com/'}
patterns={'mathjax-hosting':['off-line','offline','hard disk','server'],'mathjax-typeset':['typesetPromise','automatically','typeset()'],'uv-sync':['extra','optional'],'git-lfs-skip':['GIT_LFS_SKIP_SMUDGE'],'git-windows':['core.longpaths','MAX_PATH'],'colab-faq':['Sign in','accounts.google.com']}
records=[]
for key,url in urls.items():
 try:
  with urllib.request.urlopen(url,timeout=25) as response:raw=response.read(); final=response.url
  (s/(key+'.html')).write_bytes(raw)
  text=re.sub(r'<[^>]+>',' ',raw.decode(errors='replace')); text=re.sub(r'\s+',' ',text)
  excerpts=[]
  for term in patterns[key]:
   match=re.search(re.escape(term),text,re.I)
   if match:excerpts.append({'term':term,'excerpt':text[max(0,match.start()-130):match.end()+430]})
  records.append({'id':key,'url':url,'resolved_url':final,'accessed_at':datetime.now(UTC).isoformat(),'full_source_sha256':hashlib.sha256(raw).hexdigest(),'full_source_saved_outside_git':str(s/(key+'.html')),'excerpts':excerpts,'checked_original':True})
 except Exception as e:records.append({'id':key,'url':url,'error':str(e)})
(out/'official-current-02.json').write_text(json.dumps({'records':records},ensure_ascii=False,indent=2)+'\n');print(json.dumps(records,ensure_ascii=False))
