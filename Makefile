env_create:
	conda env create -f cv_project.yml
env_update:
	conda env update -f cv_project.yml

run_tests:
	python3 -m unittest discover


RSYNC = /usr/local/Cellar/rsync/3.1.3_1/bin/rsync
RSYNCTAGS = --archive --verbose --info=progress2 -au
HOST_FOLDER = $(HOME)/github/cv_project
REMOTE = wpq@$(COMPS0)
REMOTE_FOLDER = /u/wpq/github/cv_project

sync:
	$(RSYNC) $(RSYNCTAGS) $(HOST_FOLDER)/ $(REMOTE):$(REMOTE_FOLDER)
syncr:
	$(RSYNC) $(RSYNCTAGS) $(REMOTE):$(REMOTE_FOLDER)/ $(HOST_FOLDER)
vscode:
	sshcode --skipsync $(REMOTE) $(REMOTE_FOLDER)
