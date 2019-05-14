


env_create:
	conda env create -f cv_project.yml
env_update:
	conda env update -f cv_project.yml

run_tests:
	python3 -m unittest discover