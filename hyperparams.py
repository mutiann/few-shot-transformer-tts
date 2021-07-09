import tensorflow as tf

hparams = tf.contrib.training.HParams(
    num_mels=80,
    frame_length_ms=50,
    frame_shift_ms=12.5,
    hop_length=int(16000 * 0.0125),  # samples.
    win_length=int(16000 * 0.05),  # samples.
    max_db=100,
    ref_db=20,
    preemphasis=0.97,
    max_abs_value=4.0,
    symmetric_mel=True,
    sr=16000,
    n_fft=2048,

    n_iter=60,
    power=1.5,
    max_generation_frames=1100,
    max_eval_batches=20,
    max_eval_sample_length=1000,
    eval_sample_per_speaker=4,

    vocab_size=6000,
    embed_size=512,
    encoder_hidden=512,
    decoder_hidden=768,
    n_encoder_layer=6,
    n_decoder_layer=6,
    n_attention_head=8,
    transformer_dropout_rate=0.1,
    decoder_dropout_rate=0.5,
    prenet_hidden=256,
    postnet_hidden=512,
    n_postnet_layer=5,

    data_format="nlti",
    use_sos=True,
    bucket_size=512,
    shuffle_training_data=True,
    batch_frame_limit=8000,
    batch_frame_quad_limit=7000000,
    balanced_training=True,
    lg_prob_scale=0.2,
    adapt_start_step=30000,
    adapt_end_step=30000,
    final_adapt_rate=0.25,
    data_warmup_steps=30000,
    target_length_lower_bound=240,
    target_length_upper_bound=800,

    reg_weight=5e-9,

    multi_speaker=True,
    max_num_speaker=1000,
    speaker_embedding_size=128,

    multi_lingual=True,
    max_num_language=100,
    language_net_hidden=128,
    language_embedding_size=128,

    warmup_steps=50000,
    max_lr=1e-3,
    min_lr=1e-5,
    lr_decay_step=550000,
    lr_decay_rate=1e-2,
    adam_eps=5e-8,

    external_embed_dim=1024,
    use_external_embed=False,
)
