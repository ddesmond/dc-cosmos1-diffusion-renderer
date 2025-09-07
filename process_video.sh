CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_inverse_renderer.py \
    --checkpoint_dir checkpoints --diffusion_transformer_dir Diffusion_Renderer_Inverse_Cosmos_7B \
    --dataset_path=asset/examples/video_frames_examples/ --num_video_frames 57 --group_mode folder \
    --overlap_n_frames 32 \
    --normalize_normal 1 \
    --video_save_folder=asset/example_results/video_delighting/ \
    --offload_diffusion_transformer --offload_tokenizer