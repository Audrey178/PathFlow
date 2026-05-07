#!/usr/bin/bash

echo "*********************************"
echo "Startting experiments with baseline"
echo "*********************************"

python eval.py --config-name baseline model_path="results/new_best_accuracy/baseline.pth"

echo "Finished"
echo ""s