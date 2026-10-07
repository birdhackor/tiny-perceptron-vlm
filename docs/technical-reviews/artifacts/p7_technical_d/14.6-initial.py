import json
r=json.load(open('docs/course-experiments/results/modern.json'))['results']['variants']
for name in ['baseline','tied']:
 z=r[name]['training']['initial'];print(name,z['nll'],z['nll_sum']/z['effective_tokens'],z['effective_tokens']);print(name,'copiedoutput', 'output.weight' in r[name]['copied_initial_tables']);assert z['nll_sum']/z['effective_tokens']==z['nll']
