
GPT_CONFIG_S = {
        "context_length": 1024,
        "batch_size": 4,
        "emb_dim": 384,
        "n_layers": 6,
        "drop_rate": 0.1,
        "qkv_bias": True,
    }

GPT_CONFIG_M = {
        "context_length": 1024,
        "batch_size": 4,
        "emb_dim": 512,
        "n_layers": 6,
        "drop_rate": 0.1,
        "qkv_bias": True,
    }

GPT_CONFIG_WIKI = {
        "context_length": 1024,
        "batch_size": 8,
        "emb_dim": 512,
        "n_layers": 8,
        "drop_rate": 0.1,
        "qkv_bias": True,
    }

GPT_CONFIG_GPT2_SMALL = {
        "context_length": 1024,
        "batch_size": 4,
        "emb_dim": 768,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": True,
    }
