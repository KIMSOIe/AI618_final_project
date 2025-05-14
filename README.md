# AI618

## Running the Experiments

The models and datasets are placed in the `exp/` folder as follows:
```bash
<exp> # a folder named by the argument `--exp` given to main.py
├── datasets # all dataset files
│   ├── celeba # all CelebA files
│   ├── imagenet # all ImageNet files
│   ├── ood # out of distribution ImageNet images
│   ├── ood_bedroom # out of distribution bedroom images
│   ├── ood_cat # out of distribution cat images
│   └── ood_celeba # out of distribution CelebA images
├── logs # contains checkpoints and samples produced during training
│   ├── celeba
│   │   └── celeba_hq.ckpt # the checkpoint file for CelebA-HQ
│   ├── diffusion_models_converted
│   │   └── ema_diffusion_lsun_<category>_model
│   │       └── model-x.ckpt # the checkpoint file saved at the x-th training iteration
│   ├── imagenet # ImageNet checkpoint files
│   │   ├── 256x256_classifier.pt
│   │   ├── 256x256_diffusion.pt
│   │   ├── 256x256_diffusion_uncond.pt
│   │   ├── 512x512_classifier.pt
│   │   └── 512x512_diffusion.pt
├── image_samples # contains generated samples
└── imagenet_val_1k.txt # list of the 1k images used in ImageNet-1K.
```


## Images for Demonstration Purposes
where the following are options
- `DEGREDATION` is the type of degredation allowed. (One of: `cs2`, `cs4`, `inp`, `inp_lolcat`, `inp_lorem`, `deno`, `deblur_uni`, `deblur_gauss`, `deblur_aniso`, `sr2`, `sr4`, `sr8`, `sr16`, `sr_bicubic4`, `sr_bicubic8`, `sr_bicubic16` `color`)

- `langevin_steps`
  Number of Langevin iterations to perform during sampling.  
  (See implementation in `functions/denoising_nonlinear.py`.)

- `langevin_lr`
  Learning rate for each Langevin update step.  
  You can freely adjust this value when running inference to control step size.


### these commands can be excecuted directly:
you can freely adjust both `langevin_steps` and `langevin_lr` when running inference.

Bedroom noisy 2x without langevin:
```
python main.py --ni --config bedroom.yml --doc bedroom --timesteps 20 --eta 0.85 --etaB 1 --deg cs2 --sigma_0 0.05 -i bedroom_cs2_sigma_0.05
```


Bedroom noisy 2x with langevin:
```
python main.py --ni --config bedroom.yml --doc bedroom --timesteps 20 --eta 0.85 --etaB 1 --deg cs2 --sigma_0 0.05 -i bedroom_cs2_noise0_sigma_0.05_langevin_step10_lr_1e-4 --langevin --langevin_steps 10 --langevin_lr 1e-4
```