# The Plan(tm)

Extract q\_proj_pre for 1B and 3B vanilla each with train set

Train proc matrices on relative depths for each layer

Run the experiment with small loras for big model on q\_proj, with and without big lora for v_proj


Potential issues:

- the v_proj adapter is very necessary for performance
- The mapping matrix is so weak that the change in hidden states is minimal
  - would mean the linear mapping approach will not work
- also maybe retry with ortho proc

value dir: /pfs/work9/workspace/scratch/ka_usxcp-ws_sascha/hidden_states/1B_with_hook/self_attn.v_proj/both/scratch/slurm_tmpdir/job_1132677/1B_self_attn.v_proj_both_layer_0_self_attn_v_proj_pre.pt

key dir: /pfs/work9/workspace/scratch/ka_usxcp-ws_sascha/hidden_states/3B_with_hook/self_attn.v_proj/both/scratch/slurm_tmpdir/job_1132888/3B_self_attn.v_proj_both_layer_0_self_attn_v_proj_pre.pt

# 15.08.2025

- the experiment runs are all with non ortho procs for now
- it makes no sense to have the up and down mappings be trained with pre values only. I need the up mapping to be trained with hs just after the layer
