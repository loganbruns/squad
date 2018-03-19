EXPERIMENT=v18
DESCRIPTION="$(EXPERIMENT): architecture iteration"
WORKSPACE=main::cs224n-lbruns

all:
	@echo upload - Upload code and best_checkpoint
	@echo clean - Remove code and best_checkpoint
	@echo run-eval - Upload and run evaluation
	@echo submit-sanity - Submit to sanity test leaderboard
	@echo submit-dev - Submit to dev leaderboard

workspace:
	cl work $(WORKSPACE)

upload:
	@echo Uploading experiment $(EXPERIMENT)
	cl upload code
	cl upload experiments/$(EXPERIMENT)/best_checkpoint
	sleep 2

gen-answers: upload
	@echo Generating answers for experiment $(EXPERIMENT)
	cl run --name gen-answers-$(EXPERIMENT) --request-docker-image abisee/cs224n-dfp:v4 \
		--request-cpus 1 --request-memory 4g --request-disk 1g --request-time 1d \
		:code :best_checkpoint glove.txt:0x97c870/glove.6B.100d.txt data.json:0x4870af \
		'python code/main.py --mode=official_eval --glove_path=glove.txt --json_in_path=data.json --ckpt_load_dir=best_checkpoint'
	sleep 2
	time cl wait --tail gen-answers-$(EXPERIMENT)

run-eval: gen-answers
	@echo Running eval for experiment $(EXPERIMENT)
	cl run --name run-eval-$(EXPERIMENT) --request-docker-image abisee/cs224n-dfp:v4 \
		--request-cpus 1 --request-memory 2g --request-disk 1g --request-time 1d \
		:code data.json:0x4870af preds.json:gen-answers-$(EXPERIMENT)/predictions.json \
		'python code/evaluate.py data.json preds.json'
	sleep 2
	time cl wait --tail run-eval-$(EXPERIMENT)

submit-sanity:
	cl edit -w $(WORKSPACE) gen-answers-$(EXPERIMENT) -T cs224n-win18-sanity-check --description $(DESCRIPTION)

submit-dev:
	cl edit -w $(WORKSPACE) gen-answers-$(EXPERIMENT) -T cs224n-win18-dev --description $(DESCRIPTION)

clean:
	cl detach -w $(WORKSPACE) run-eval-$(EXPERIMENT) gen-answers-$(EXPERIMENT) code best_checkpoint

local-sanity:
	python code/main.py --mode=official_eval \
		--json_in_path=data/tiny-dev.json \
		--ckpt_load_dir=experiments/$(EXPERIMENT)/best_checkpoint
	python code/evaluate.py data/tiny-dev.json predictions.json

local-dev:
	python code/main.py --mode=official_eval \
		--json_in_path=data/dev-v1.1.json \
		--ckpt_load_dir=experiments/$(EXPERIMENT)/best_checkpoint
	python code/evaluate.py data/dev-v1.1.json predictions.json

show-examples:
	python code/main.py --experiment_name=$(EXPERIMENT) --mode=show_examples

train:
	python code/main.py --experiment_name=$(EXPERIMENT) --mode=train

train-cpu:
	python code/main.py --experiment_name=$(EXPERIMENT) --mode=train --gpu=-1

train-slow:
	python code/main.py --experiment_name=$(EXPERIMENT) --mode=train --learning_rate=0.0005 --max_gradient_norm=3.5

train-very-slow:
	python code/main.py --experiment_name=$(EXPERIMENT) --mode=train --learning_rate=0.0001 --max_gradient_norm=2.5

tensorboard:
	(cd experiments; nohup tensorboard --logdir=. --port=5678 &)
