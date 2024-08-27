# MoE Analysis
## Usage
```sh
# prepare
jq --raw-output '.[] | select(.documents | tonumber | .>=10000) | .code' ../gpt/culturax/culturax-v1-0-0_data_stats.json >languages.txt
# submit the slurm job
./noctua2_run <unique batch name> <path to the model checkpoint>
# ...
# prepare the token summary table
./make_lang_experts --dir=datadir
# plot the expert-language heatmap
./heatmap.gnuplot datadir/lang_experts.norm.sorted.image datadir/lang_experts.norm.sorted.png "title"
# sort experts inside layers and also generate expert specificity table
./sort_image --dir=datadir
# plot the expert specificity figure
./plot_expert_metric datadir/lang_experts.max.dat datadir/lang_experts.max.png
# language classification
./run_lang_tsne --dir=datadir
./plot_lang_tsne datadir
# expert classification
./run_expert_tsne --dir=datadir
./plot_expert_tsne datadir
```
