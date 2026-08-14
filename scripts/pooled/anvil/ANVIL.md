# Running pooled ONE-NAS on ACCESS Anvil

## One-time setup
1. Push the `pooled-panel` branch to your GitHub fork, then on Anvil:
   ```
   git clone -b pooled-panel https://github.com/<you>/ONENAS.git ~/ONENAS
   cd ~/ONENAS && module load gcc openmpi cmake
   mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j 16
   ```
2. Upload the prepared panels from your Mac:
   ```
   rsync -av ~/.claude/jobs/fc17658e/tmp/set{1,2,3,4}_clean anvil:~/panels/
   ```
3. Find your allocation ID (`myallocations`) and put it in the sbatch `-A` line.

## Submitting
- Vanilla (Uniform) arm, all 4 sets:  `sbatch --array=1-4 vanilla_onenas.sbatch`
- PER arm: same script with `--export=ALL,SAMPLER=PER` once the PER variant
  script lands (or copy the script and change --get_train_data_by + PER params).
- Extra seeds: `sbatch --array=1-4 --export=ALL,SEED=43 vanilla_onenas.sbatch`

## Notes
- Anvil `shared` partition: up to 128 cores/node, jobs share nodes; 16 tasks
  (1 master + 15 workers) is a good starting size — scale -n up once timed.
- Wall time: the Mac does ~2 min/generation on 4 workers; 15 workers should be
  well under 12h for 300 generations. Check the first job's timing and tighten.
- Scoring runs anywhere (pure Python): `scripts/pooled/score_stream.py`.
- Do NOT mix flags from older ONE-NAS scripts (e.g. --online_learning,
  --num_generations, --sequence_length, --train_with): this fork's CLI differs.
