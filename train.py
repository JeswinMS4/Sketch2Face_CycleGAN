"""
Training for CycleGAN
"""

import torch
from dataset import PhotoSketchDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator


def train_fn(
    disc_photo, disc_sketch, gen_sketch, gen_photo, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler
):
    photo_reals = 0
    photo_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (photo, sketch) in enumerate(loop):
        photo = photo.to(config.DEVICE)
        sketch = sketch.to(config.DEVICE)

        # Train Discriminators Photo and Sketch
        with torch.cuda.amp.autocast():
            fake_sketch = gen_sketch(photo)
            D_photo_real = disc_photo(photo)
            D_photo_fake = disc_photo(fake_sketch.detach())
            photo_reals += D_photo_real.mean().item()
            photo_fakes += D_photo_fake.mean().item()
            D_photo_real_loss = mse(D_photo_real, torch.ones_like(D_photo_real))
            D_photo_fake_loss = mse(D_photo_fake, torch.zeros_like(D_photo_fake))
            D_photo_loss = D_photo_real_loss + D_photo_fake_loss

            fake_photo = gen_photo(sketch)
            D_sketch_real = disc_sketch(sketch)
            D_sketch_fake = disc_sketch(fake_photo.detach())
            D_sketch_real_loss = mse(D_sketch_real, torch.ones_like(D_sketch_real))
            D_sketch_fake_loss = mse(D_sketch_fake, torch.zeros_like(D_sketch_fake))
            D_sketch_loss = D_sketch_real_loss + D_sketch_fake_loss

            # Combine the losses
            D_loss = (D_photo_loss + D_sketch_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators Photo and Sketch
        with torch.cuda.amp.autocast():
            # Adversarial loss for both generators
            D_photo_fake = disc_photo(fake_sketch)
            D_sketch_fake = disc_sketch(fake_photo)
            loss_G_photo = mse(D_photo_fake, torch.ones_like(D_photo_fake))
            loss_G_sketch = mse(D_sketch_fake, torch.ones_like(D_sketch_fake))

            # Cycle loss
            cycle_photo = gen_photo(fake_sketch)
            cycle_sketch = gen_sketch(fake_photo)
            cycle_photo_loss = l1(photo, cycle_photo)
            cycle_sketch_loss = l1(sketch, cycle_sketch)

            # Identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_photo = gen_photo(photo)
            identity_sketch = gen_sketch(sketch)
            identity_photo_loss = l1(photo, identity_photo)
            identity_sketch_loss = l1(sketch, identity_sketch)

            # Combine all losses
            G_loss = (
                loss_G_sketch
                + loss_G_photo
                + cycle_sketch_loss * config.LAMBDA_CYCLE
                + cycle_photo_loss * config.LAMBDA_CYCLE
                + identity_photo_loss * config.LAMBDA_IDENTITY
                + identity_sketch_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_image(photo*0.5+0.5, f"saved_images/real_photo_{idx}.png")
            save_image(fake_photo*0.5+0.5, f"saved_images/sketch_to_photo_{idx}.png")
            save_image(sketch*0.5+0.5, f"saved_images/real_sketch_{idx}.png")
            save_image(fake_sketch*0.5+0.5, f"saved_images/photo_to_sketch_{idx}.png")

        loop.set_postfix(Photo_real=photo_reals / (idx + 1), Photo_fake=photo_fakes / (idx + 1))


def main():
    disc_photo = Discriminator(in_channels=3).to(config.DEVICE)
    disc_sketch = Discriminator(in_channels=3).to(config.DEVICE)
    gen_sketch = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_photo = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_photo.parameters()) + list(disc_sketch.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_sketch.parameters()) + list(gen_photo.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_PHOTO,
            gen_photo,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_SKETCH,
            gen_sketch,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_PHOTO,
            disc_photo,
            opt_disc,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_SKETCH,
            disc_sketch,
            opt_disc,
            config.LEARNING_RATE,
        )

    dataset = PhotoSketchDataset(
        root_photo=config.TRAIN_DIR + "/photos",
        root_sketch=config.TRAIN_DIR + "/sketches",
        transform=config.transforms,
    )
    val_dataset = PhotoSketchDataset(
        root_photo=config.VAL_DIR+"/photos",
        root_sketch=config.VAL_DIR+"/sketches",
        transform=config.transforms,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        print("EPOCH == ",epoch)
        train_fn(
            disc_photo,
            disc_sketch,
            gen_sketch,
            gen_photo,
            loader,
            opt_disc,
            opt_gen,
            L1,
            mse,
            d_scaler,
            g_scaler,
        )

        if config.SAVE_MODEL:
            save_checkpoint(gen_photo, opt_gen, filename=config.CHECKPOINT_GEN_PHOTO)
            save_checkpoint(gen_sketch, opt_gen, filename=config.CHECKPOINT_GEN_SKETCH)
            save_checkpoint(disc_photo, opt_disc, filename=config.CHECKPOINT_CRITIC_PHOTO)
            save_checkpoint(disc_sketch, opt_disc, filename=config.CHECKPOINT_CRITIC_SKETCH)


if __name__ == "__main__":
    main()
