CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_inverse_renderer.py \
    --checkpoint_dir checkpoints --diffusion_transformer_dir Diffusion_Renderer_Inverse_Cosmos_7B \
    --dataset_path=asset/examples/image_examples/ --num_video_frames 1 --group_mode webdataset \
    --video_save_folder=asset/example_results/image_delighting/ --save_video=False