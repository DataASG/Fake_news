# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* Fake_news/*.py

black:
	@black scripts/* Fake_news/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit=$(VIRTUAL_ENV)/lib/python*

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr Fake_news-*.dist-info
	@rm -fr Fake_news.egg-info

install:
	@pip install . -U

all: clean install test black check_code


uninstal:
	@python setup.py install --record files.txt
	@cat files.txt | xargs rm -rf
	@rm -f files.txt

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u lologibus2

pypi:
	@twine upload dist/* -u lologibus2



##### Machine Type - - - - - - - - - - - - - - - - - - - - - - - - -
MACHINE_TYPE=n1-highmem-16
BUCKET_NAME=fakenews475
BUCKET_TRAINING_FOLDER=trainings
REGION=europe-west1
PYTHON_VERSION=3.7
PACKAGE_NAME=Fake_news
FILENAME=main
RUNTIME_VERSION=2.1
JOB_NAME=fake_news_training_pipeline_2k_validation_1_layer$(shell date +'%Y%m%d_%H%M%S')
##### Machine Type - - - - - - - - - - - - - - - - - - - - - - - - -
MACHINE_TYPE=n1-highmem-96  #complex_model_l_gpu old

gcp_submit_training:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
		--job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER} \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
		--scale-tier CUSTOM \
		--master-machine-type ${MACHINE_TYPE}



run_locally:
	@python -m ${PACKAGE_NAME}.${FILENAME}


