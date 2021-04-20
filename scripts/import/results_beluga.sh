which_beluga=$1
#!/bin/bash
rsync -avm --include='*.npy' --include='*/' --exclude='*' \
            ${which_beluga}@beluga.computecanada.ca:~/scratch/histo_fsl/checkpoints ./
