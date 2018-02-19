EXPERIMENT=baseline
DESCRIPTION="Baseline test"

all:
	@echo upload - Upload code and experiment
	@echo run-eval - Run evaluation
	@echo submit-sanity-check - Run evaluation

workspace:
	cl work main::cs224n-lbruns

upload:
	@echo Uploading experiment $(EXPERIMENT)
	cl upload code
	cl upload experiments/$(EXPERIMENT)/best_checkpoint

gen-answers: upload
	@echo Generating answers for experiment $(EXPERIMENT)
	cl run --name gen-answers --request-docker-image abisee/cs224n-dfp:v4 :code :best_checkpoint glove.txt:0x97c870/glove.6B.100d.txt data.json:0x4870af \
		'python code/main.py --mode=official_eval --glove_path=glove.txt --json_in_path=data.json --ckpt_load_dir=best_checkpoint'
	time cl wait --tail gen-answers

run-eval: gen-answers
	@echo Running eval for experiment $(EXPERIMENT)
	cl run --name run-eval --request-docker-image abisee/cs224n-dfp:v4 :code data.json:0x4870af preds.json:gen-answers/predictions.json \
		'python code/evaluate.py data.json preds.json'	
	time cl wait --tail run-eval-$(EXPERIMENT)

submit-sanity-check:
	cl edit gen-answers -T cs224n-win18-sanity-check --description $(DESCRIPTION)

submit-dev:
	cl edit gen-answers -T cs224n-win18-dev --description $(DESCRIPTION)

clean:
	cl rm run-eval gen-answers code best_checkpoint
