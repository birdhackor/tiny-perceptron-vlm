import sys,json,unittest.mock,torch
from scripts.course_experiments import run
spec=run.experiment_spec('posttraining');print('original_dispatch_spec',spec['module'],spec['function'])
with unittest.mock.patch.object(run,'execute',side_effect=RuntimeError('review parser stop before training')) as mocked:
 sys.argv=['run','--experiment','posttraining','--device','cpu']
 try:run.main()
 except RuntimeError as e:print(str(e))
 print('parsed_arguments',mocked.call_args.args[:2],mocked.call_args.kwargs)
x=json.load(open('docs/course-experiments/results/posttraining.json'));print('raw_times_total',x['elapsed_seconds'],'inner',x['results']['experiment_seconds'],'ppo',x['results']['ppo']['seconds']);print('raw_environment',x['device'],x['torch_version'],x['python_version']);print('torch',torch.__version__)
