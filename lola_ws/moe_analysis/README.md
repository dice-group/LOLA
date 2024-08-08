# MoE Analysis
## Usage
```
jq --raw-output '.[] | select(.documents | tonumber | .>=10000) | .code' ../gpt/culturax/culturax-v1-0-0_data_stats.json >languages.txt
```

```
./noctua2_run
```
