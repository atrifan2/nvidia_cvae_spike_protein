#!/bin/bash -l

function usage()   {
	echo "USAGE: $0 < lab | notebook >"
	exit 1
}

case $1 in
	lab)
		jupyter_type=lab;;
	notebook)
		jupyter_type=notebook;;
	*)
		usage;;
esac

jupyter $jupyter_type \
--no-browser \
--port=8888 \
--ip=0.0.0.0 \
--NotebookApp.notebook_dir="/projects" \
--NotebookApp.password=\"\" \
--NotebookApp.token=\"\" \
--NotebookApp.password_required=False
