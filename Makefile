# Optional convenience only. Everything here is a single python command and
# the README gives each of them, so make is never required.

PYTHON ?= python

.PHONY: help reproduce analysis cox verify crosscheck selftest hygiene clean distclean

help:
	@echo "reproduce   download the data if needed, run the analysis, verify it"
	@echo "analysis    the replication itself, about 10 seconds"
	@echo "cox         age- and sex-adjusted Cox models, about 40 seconds"
	@echo "selftest    correctness checks on the Cox implementation only"
	@echo "verify      assert every number quoted in README.md, plus hygiene"
	@echo "hygiene     repository hygiene checks only, no data needed"
	@echo "crosscheck  optional, needs requirements.txt installed"
	@echo "clean       remove generated results"
	@echo "distclean   also remove the downloaded dataset"

reproduce: analysis cox verify

analysis:
	$(PYTHON) ecg_age_bias.py

cox:
	$(PYTHON) cox_model.py

selftest:
	$(PYTHON) cox_model.py --selftest

verify:
	$(PYTHON) verify.py

hygiene:
	$(PYTHON) verify.py --hygiene

crosscheck:
	$(PYTHON) crosscheck_lifelines.py

clean:
	-rm -f results/*.json
	-rm -rf __pycache__

distclean: clean
	-rm -f data/exams.csv data/exams.csv.part
