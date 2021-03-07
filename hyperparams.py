import tensorflow as tf

hparams = tf.contrib.training.HParams(
    # Audio
    num_mels=80,
    # num_freq = 1024
    n_fft=2048,
    sr=16000,
    # frame_length_ms = 50.
    # frame_shift_ms = 12.5
    preemphasis=0.97,
    frame_shift=0.0125,  # seconds
    frame_length=0.05,  # seconds
    hop_length=int(16000 * 0.0125),  # samples.
    win_length=int(16000 * 0.05),  # samples.
    n_mels=80,  # Number of Mel banks to generate
    power=1.2,  # Exponent for amplifying the predicted magnitude
    min_level_db=-100,
    ref_level_db=20,
    ref_amplitude=0.14848, # 95% max amplitude of LJSpeech
    enc_hidden_size=512,
    dec_hidden_size=768,
    postnet_hidden=512,
    use_spk_lang=True,
    embedding_size=512,
    num_heads=8,
    num_layers=6,
    stop_token_weight=1.0,
    stop_token_weight_anneal_start=80000,
    stop_token_weight_anneal_step=160000,
    max_db=100,
    ref_db=20,

    n_iter=60,
    # power = 1.5,
    outputs_per_step=1,

    lr_max=1e-4,
    lr_decay_steps=850000,
    lr_min=1e-5,
    warmup_steps=60000,
    lr_decay_type='noam',

    save_step=10000,
    image_step=2500,

    batch_frame_limit=6000,
    batch_frame_quad_limit=5000000,

    filter_length=True,
    filter_length_step=60000
)
