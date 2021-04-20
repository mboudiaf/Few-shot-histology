#!/bin/bash
rsync -av --exclude plots \
          --exclude checkpoints/ \
          --exclude logs/ \
          --exclude notebooks/ \
	  --exclude results/ \
          --exclude figures/ \
          --exclude data \
          --exclude .git \
          --exclude *.sublime-project \
          --exclude *.sublime-workspace \
          --exclude __pycache__ \
           ./ mboudiaf@beluga.computecanada.ca:~/scratch/histo_fsl/
