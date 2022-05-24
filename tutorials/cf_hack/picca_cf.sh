#!/bin/bash -l

#SBATCH -C haswell
#SBATCH --partition=regular
#SBATCH --account=desi
#SBATCH --nodes=8
#SBATCH --time=02:00:00
#SBATCH --job-name=cf
#SBATCH --output=/global/cfs/cdirs/desi/users/acuceu/run/picca_on_mocks/desi2/min_brightness/logs/cf-%j.out
#SBATCH --error=/global/cfs/cdirs/desi/users/acuceu/run/picca_on_mocks/desi2/min_brightness/logs/cf-%j.err

module load python
conda activate /global/common/software/desi/users/acuceu/conda/picca

# umask 0002
# export OMP_NUM_THREADS=32

# command="picca_cf.py --in-dir /global/cfs/cdirs/desi/users/acuceu/run/picca_on_mocks/desi2/min_brightness/deltas_LYA/ --out /global/cfs/cdirs/desi/users/acuceu/run/picca_on_mocks/desi2/min_brightness/correlations/cf.fits.gz --nproc 32"

# echo $command

# srun -N 1 -n 1 -c 32 $command

# piccaenv

umask 0002
export OMP_NUM_THREADS=32

# Write the number of nodes here!
numnodes=8

files=`ls -1 /global/cfs/cdirs/desi/users/acuceu/run/picca_on_mocks/desi2/min_brightness/deltas_LYA/delta-*.fits*`
nfiles=`echo $files | wc -w`
nfilespernode=$(( $nfiles/$numnodes + 1))

first=1
last=$nfilespernode
for node in `seq $numnodes` ; do
    echo "starting node $node"

    # list of files to run
    if (( $node == $numnodes )) ; then
	last=""
    fi
    echo ${first}-${last}
    # tfiles=`echo $files | cut -d " " -f ${first}-${last}`
    first=$(( $first + $nfilespernode ))
    last=$(( $last + $nfilespernode ))
    command="srun -N 1 -n 1 -c 32 python /global/homes/a/acuceu/lib/picca_dev/picca/bin/picca_cf_hack.py --in-dir /global/cfs/cdirs/desi/users/acuceu/run/picca_on_mocks/desi2/min_brightness/deltas_LYA/ --out /global/cfs/cdirs/desi/users/acuceu/run/picca_on_mocks/desi2/min_brightness/correlations/cf2/cf_node_$node.p --node-ind $node --num-nodes $numnodes --nproc 32"
    echo $command
    $command >& /global/cfs/cdirs/desi/users/acuceu/run/picca_on_mocks/desi2/min_brightness/logs2/cf-node-$node.log &
done

wait
echo "END"


# Only run this part once everything above is finished!!!
# The --in-dir-node option should be the same path as --out above, where each node writes it's own subset results
# --num-nodes should match how many output files there are (nodes used above)
command="srun -N 1 -n 1 -c 32 python /global/homes/a/acuceu/lib/picca_dev/picca/bin/picca_cf_collect.py --in-dir /global/cfs/cdirs/desi/users/acuceu/run/picca_on_mocks/desi2/min_brightness/deltas_LYA/ --in-dir-node /global/cfs/cdirs/desi/users/acuceu/run/picca_on_mocks/desi2/min_brightness/correlations/cf2/ --out /global/cfs/cdirs/desi/users/acuceu/run/picca_on_mocks/desi2/min_brightness/correlations/cf2.fits.gz --node-ind 1 --num-nodes $numnodes --nproc 32"
$command
