# The Plan(tm)

- [ ] Retrainings of Lora models, just to be sure that they are all properly on train and val set
  - [ ] 1B
    - [ ] Untied QV
    - [x] Tied QV
    - [x] Tied Q
  - [ ] 3B
    - [ ] Untied QV
    - [x] Tied QV
    - [x] Tied Q
- [ ] Run Eval on Test set
  - [ ] 1B
    - [ ] Untied QV
    - [ ] Tied QV
    - [ ] Tied Q
  - [ ] 3B
    - [ ] Untied QV
    - [ ] Tied QV
    - [ ] Tied Q

Value dir Q pre
/pfs/work9/workspace/scratch/ka_usxcp-ws_sascha/hidden_states/1B_with_hook/self_attn.q_proj
Key dir Q pre
/pfs/work9/workspace/scratch/ka_usxcp-ws_sascha/hidden_states/3B_with_hook/self_attn.q_proj/pre/scratch/slurm_tmpdir/job_1103951

Value dir Q Post
/pfs/work9/workspace/scratch/ka_usxcp-ws_sascha/hidden_states/1B_with_hook/self_attn.q_proj/post/scratch/slurm_tmpdir/job_1143288
Key dir Q Post
/pfs/work9/workspace/scratch/ka_usxcp-ws_sascha/hidden_states/3B_with_hook/self_attn.q_proj/post/scratch/slurm_tmpdir/job_1143289

Value dir V Pre
/pfs/work9/workspace/scratch/ka_usxcp-ws_sascha/hidden_states/1B_with_hook/self_attn.v_proj/pre
Key dir V Pre
/pfs/work9/workspace/scratch/ka_usxcp-ws_sascha/hidden_states/3B_with_hook/self_attn.v_proj/pre

Value dir V Post
/pfs/work9/workspace/scratch/ka_usxcp-ws_sascha/hidden_states/1B_with_hook/self_attn.v_proj/post
Key dir V Post
/pfs/work9/workspace/scratch/ka_usxcp-ws_sascha/hidden_states/3B_with_hook/self_attn.v_proj/post

# Hemmingway bridge

https://ai.plainenglish.io/decoding-an-llms-thoughts-logit-lens-in-just-25-lines-of-code-100c1dbf2ac0