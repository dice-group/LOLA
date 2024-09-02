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

DATADIR=datadir/
# sort experts inside layers and also generate expert specificity table
./sort_image --dir=${DATADIR}
# plot the expert-language heatmap
./heatmap.gnuplot ${DATADIR}lang_experts.norm.image ${DATADIR}lang_experts.norm.png "${DATADIR}"
./heatmap.gnuplot ${DATADIR}lang_experts.norm.sorted.image ${DATADIR}lang_experts.norm.sorted.png "sorted ${DATADIR}"
# plot the expert specificity figure
./plot_expert_metric ${DATADIR}lang_experts.max.dat ${DATADIR}/lang_experts.max.png
# language classification
./run_lang_tsne --dir=${DATADIR}
./plot_lang_tsne ${DATADIR}
# expert classification
./run_lang_tsne --dir=${DATADIR}
./plot_lang_tsne ${DATADIR}

convert */lang_experts.norm.png -set delay 100 lang_experts.norm.gif
convert lang_experts.norm.gif \( -clone -1 -set delay 300 \) -swap -2,-1 +delete lang_experts.norm.2.gif
```
