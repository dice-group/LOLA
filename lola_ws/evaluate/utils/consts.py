# Note: the model ids in the sets below are made by conjoining huggingface organization and model name together with __ (similar to how lm-eval-harness does it)
DEFAULT_MODELS = {'dice-research__lola_v1'}
# Models to exclude
EXCLUDED_MODELS = {'SeaLLMs__SeaLLMs-v3-1.5B-Chat', 'facebook__m2m100_1.2B'}
# All the incompatible task and language pairs
INCOMPATIBLE_SUBTASKS_MODELS = {('belebele','facebook__mbart-large-50'),
                                ('belebele','google__mt5-large'),
                                ('belebele','google__mt5-xl'),
                                ('belebele','google__umt5-xl')}