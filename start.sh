#!/bin/bash

if [ "$1" == "worker" ]; then
	echo "Starting worker node"
	ray start --address="$2:6379"
else
	echo "Starting worker head"
	ray start --head --node-ip-address="$2" --port=6379
fi
