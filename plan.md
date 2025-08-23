# The Plan(tm)

Extract q\_proj_pre for 1B and 3B vanilla each with train set

Train proc matrices on relative depths for each layer

Run the experiment with small loras for big model on q\_proj, with and without big lora for v_proj


Potential issues:

- the v_proj adapter is very necessary for performance
- The mapping matrix is so weak that the change in hidden states is minimal
  - would mean the linear mapping approach will not work
- also maybe retry with ortho proc

value dir: /pfs/work9/workspace/scratch/ka_usxcp-ws_sascha/hidden_states/1B_with_hook/self_attn.q_proj/post/scratch/slurm_tmpdir/job_1143288

key dir: /pfs/work9/workspace/scratch/ka_usxcp-ws_sascha/hidden_states/3B_with_hook/self_attn.q_proj/post/scratch/slurm_tmpdir/job_1143289

# Hemmingway bridge

Fix mapping loading to handle different mappings per module
