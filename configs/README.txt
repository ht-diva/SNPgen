We need to load every base.yaml file and then the specific configuration.
For instance:
    ['./configs/vae/base.yaml', './configs/vae/encoder/base.yaml', './configs/vae/encoder/small_emb64.yaml']

To use the discriminator:
    ['./configs/vae/base.yaml', './configs/vae/encoder/base.yaml', './configs/vae/encoder/small_emb64.yaml', './configs/vae/base_disc.yaml']