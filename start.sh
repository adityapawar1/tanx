#!/bin/bash

if [ "$1" == "worker" ]; then
	echo "Starting worker node"
	RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1 ray start --address="$2:6379"
else
	echo "Starting head node"
	ray start --head --node-ip-address="$2" --port=6379 --num-cpus=2
fi
