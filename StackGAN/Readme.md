May need to export PYTHONPATH=path_to_StackGAN

In StackGAN/

put the scene sentences in Data/birds/scene_text.txt

$ CUDA_VISIBLE_DEVICES=0 ./demo/birds_demo.sh Data/birds/scene_text

The output will be in Data/birds/scene_text/result%d.jpg
  %d = 0, 1, 2 ..., the order is the same as the sentences
