#!/usr/bin/env python3
"""
AML Lab - Practical 12
Basic Generative Adversarial Network (GAN) using MNIST.

Course Outcome: CO5
Framework: TensorFlow / Keras
Execution: CPU-compatible
"""

import argparse
import logging
import os
import random
import time
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers


# ============================================================
# 01. CONFIGURATION
# ============================================================

SEED = 42

BUFFER_SIZE = 60000
BATCH_SIZE = 256
NOISE_DIM = 100
EPOCHS = 50
NUM_EXAMPLES_TO_GENERATE = 16

LEARNING_RATE = 1e-4

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

GENERATED_DIR = OUTPUT_DIR / "generated_images"
PLOTS_DIR = OUTPUT_DIR / "plots"
MODELS_DIR = OUTPUT_DIR / "models"
LOGS_DIR = OUTPUT_DIR / "logs"

for directory in (
    GENERATED_DIR,
    PLOTS_DIR,
    MODELS_DIR,
    LOGS_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# 02. REPRODUCIBILITY AND LOGGING
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "training.log", mode="a"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("MNIST-GAN")


# ============================================================
# 03. DATASET LOADING AND PREPROCESSING
# ============================================================

def load_mnist():
    """Load and normalize the MNIST training dataset."""

    logger.info("Loading MNIST dataset...")

    (x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32")

    # Normalize pixel values from [0, 255] to [-1, 1].
    x_train = (x_train - 127.5) / 127.5

    # Add grayscale channel dimension.
    x_train = np.expand_dims(x_train, axis=-1)

    logger.info("Dataset shape: %s", x_train.shape)
    logger.info("Pixel range: %.2f to %.2f",
                float(x_train.min()), float(x_train.max()))

    dataset = (
        tf.data.Dataset.from_tensor_slices(x_train)
        .shuffle(BUFFER_SIZE, seed=SEED)
        .batch(BATCH_SIZE)
        .prefetch(1)
    )

    return x_train, dataset


# ============================================================
# 04. GENERATOR MODEL
# ============================================================

def build_generator():
    """Build the generator that converts random noise into images."""

    model = tf.keras.Sequential(
        [
            layers.Input(shape=(NOISE_DIM,)),

            layers.Dense(7 * 7 * 256, use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),

            layers.Reshape((7, 7, 256)),

            layers.Conv2DTranspose(
                128,
                kernel_size=5,
                strides=1,
                padding="same",
                use_bias=False,
            ),
            layers.BatchNormalization(),
            layers.LeakyReLU(),

            layers.Conv2DTranspose(
                64,
                kernel_size=5,
                strides=2,
                padding="same",
                use_bias=False,
            ),
            layers.BatchNormalization(),
            layers.LeakyReLU(),

            layers.Conv2DTranspose(
                1,
                kernel_size=5,
                strides=2,
                padding="same",
                use_bias=False,
                activation="tanh",
            ),
        ],
        name="Generator",
    )

    return model


# ============================================================
# 05. DISCRIMINATOR MODEL
# ============================================================

def build_discriminator():
    """Build the discriminator that classifies real and fake images."""

    model = tf.keras.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),

            layers.Conv2D(
                64,
                kernel_size=5,
                strides=2,
                padding="same",
            ),
            layers.LeakyReLU(),
            layers.Dropout(0.3),

            layers.Conv2D(
                128,
                kernel_size=5,
                strides=2,
                padding="same",
            ),
            layers.LeakyReLU(),
            layers.Dropout(0.3),

            layers.Flatten(),
            layers.Dense(1),
        ],
        name="Discriminator",
    )

    return model


# ============================================================
# 06. LOSS FUNCTIONS AND OPTIMIZERS
# ============================================================

cross_entropy = tf.keras.losses.BinaryCrossentropy(
    from_logits=True
)


def discriminator_loss(real_output, fake_output):
    """Calculate discriminator loss."""

    real_loss = cross_entropy(
        tf.ones_like(real_output),
        real_output,
    )

    fake_loss = cross_entropy(
        tf.zeros_like(fake_output),
        fake_output,
    )

    return real_loss + fake_loss


def generator_loss(fake_output):
    """Calculate generator loss."""

    return cross_entropy(
        tf.ones_like(fake_output),
        fake_output,
    )


# ============================================================
# 07. TRAINING STEP
# ============================================================

@tf.function(reduce_retracing=True)
def train_step(
    images,
    generator,
    discriminator,
    generator_optimizer,
    discriminator_optimizer,
):
    """Train both networks for one batch."""

    # Dynamic batch size prevents noise/image dimension mismatch.
    current_batch_size = tf.shape(images)[0]

    noise = tf.random.normal(
        [current_batch_size, NOISE_DIM]
    )

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        generated_images = generator(
            noise,
            training=True,
        )

        real_output = discriminator(
            images,
            training=True,
        )

        fake_output = discriminator(
            generated_images,
            training=True,
        )

        gen_loss = generator_loss(fake_output)

        disc_loss = discriminator_loss(
            real_output,
            fake_output,
        )

    generator_gradients = gen_tape.gradient(
        gen_loss,
        generator.trainable_variables,
    )

    discriminator_gradients = disc_tape.gradient(
        disc_loss,
        discriminator.trainable_variables,
    )

    generator_optimizer.apply_gradients(
        zip(
            generator_gradients,
            generator.trainable_variables,
        )
    )

    discriminator_optimizer.apply_gradients(
        zip(
            discriminator_gradients,
            discriminator.trainable_variables,
        )
    )

    return gen_loss, disc_loss


# ============================================================
# 08. IMAGE GENERATION AND VISUALIZATION
# ============================================================

def save_generated_images(
    generator,
    epoch,
    seed,
    final=False,
):
    """Generate and save a 4x4 grid of synthetic digits."""

    predictions = generator(
        seed,
        training=False,
    )

    predictions = (predictions + 1.0) / 2.0
    predictions = tf.clip_by_value(
        predictions,
        0.0,
        1.0,
    )

    fig, axes = plt.subplots(
        4,
        4,
        figsize=(8, 8),
    )

    for index, axis in enumerate(axes.flat):
        axis.imshow(
            predictions[index, :, :, 0].numpy(),
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axis.axis("off")

    title = (
        "Final Generated MNIST Images"
        if final
        else f"Generated Images - Epoch {epoch}"
    )

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if final:
        filename = GENERATED_DIR / "final_generated_images.png"
    else:
        filename = GENERATED_DIR / f"epoch_{epoch:03d}.png"

    fig.savefig(
        filename,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    logger.info("Saved generated images: %s", filename)


# ============================================================
# 09. LOSS GRAPH
# ============================================================

def save_loss_plot(history):
    """Plot generator and discriminator training losses."""

    fig, axis = plt.subplots(figsize=(10, 5))

    axis.plot(
        history["epoch"],
        history["generator_loss"],
        label="Generator Loss",
    )

    axis.plot(
        history["epoch"],
        history["discriminator_loss"],
        label="Discriminator Loss",
    )

    axis.set_title("GAN Training Loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.legend()
    axis.grid(True, alpha=0.3)

    fig.tight_layout()

    filename = PLOTS_DIR / "gan_training_loss.png"

    fig.savefig(
        filename,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    logger.info("Saved loss graph: %s", filename)


# ============================================================
# 10. COMPLETE TRAINING PROCESS
# ============================================================

def train_gan(epochs=EPOCHS, limit_samples=None):
    """Execute the complete GAN training process."""

    if epochs < 1:
        raise ValueError("Epochs must be at least 1.")

    if limit_samples is not None and limit_samples < 1:
        raise ValueError("Sample limit must be at least 1.")

    logger.info("=" * 60)
    logger.info("PRACTICAL 12: BASIC GAN USING MNIST")
    logger.info("=" * 60)

    logger.info("TensorFlow version: %s", tf.__version__)
    logger.info("Epochs: %d", epochs)
    logger.info("Batch size: %d", BATCH_SIZE)
    logger.info("Noise dimension: %d", NOISE_DIM)
    logger.info("CPU devices: %s", tf.config.list_physical_devices("CPU"))
    logger.info("GPU devices: %s", tf.config.list_physical_devices("GPU"))

    x_train, dataset = load_mnist()

    if limit_samples is not None:
        x_train = x_train[:limit_samples]

        dataset = (
            tf.data.Dataset.from_tensor_slices(x_train)
            .shuffle(
                len(x_train),
                seed=SEED,
            )
            .batch(BATCH_SIZE)
            .prefetch(1)
        )

        logger.info(
            "Limited training samples: %d",
            len(x_train),
        )

    generator = build_generator()
    discriminator = build_discriminator()

    generator.summary(print_fn=logger.info)
    discriminator.summary(print_fn=logger.info)

    generator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )

    discriminator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )

    seed = tf.random.normal(
        [NUM_EXAMPLES_TO_GENERATE, NOISE_DIM]
    )

    history = []

    logger.info("Starting GAN training...")

    training_start = time.time()

    for epoch in range(1, epochs + 1):

        epoch_start = time.time()

        generator_loss_values = []
        discriminator_loss_values = []

        for batch_images in dataset:

            gen_loss, disc_loss = train_step(
                batch_images,
                generator,
                discriminator,
                generator_optimizer,
                discriminator_optimizer,
            )

            generator_loss_values.append(
                float(gen_loss.numpy())
            )

            discriminator_loss_values.append(
                float(disc_loss.numpy())
            )

        average_gen_loss = float(
            np.mean(generator_loss_values)
        )

        average_disc_loss = float(
            np.mean(discriminator_loss_values)
        )

        epoch_duration = time.time() - epoch_start

        history.append(
            {
                "epoch": epoch,
                "generator_loss": average_gen_loss,
                "discriminator_loss": average_disc_loss,
                "duration_seconds": epoch_duration,
            }
        )

        logger.info(
            "Epoch %d/%d | Generator Loss: %.4f | "
            "Discriminator Loss: %.4f | Time: %.2f sec",
            epoch,
            epochs,
            average_gen_loss,
            average_disc_loss,
            epoch_duration,
        )

        if epoch % 10 == 0 or epoch == epochs:
            save_generated_images(
                generator,
                epoch,
                seed,
            )

    total_duration = time.time() - training_start

    # Save final generated image grid.
    save_generated_images(
        generator,
        epochs,
        seed,
        final=True,
    )

    # Save model files.
    generator.save(
        MODELS_DIR / "generator.keras"
    )

    discriminator.save(
        MODELS_DIR / "discriminator.keras"
    )

    # Save training history.
    history_df = pd.DataFrame(history)

    history_file = LOGS_DIR / "training_history.csv"

    history_df.to_csv(
        history_file,
        index=False,
    )

    save_loss_plot(history_df)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETED")
    logger.info("Total training time: %.2f seconds", total_duration)
    logger.info("Generator model: %s", MODELS_DIR / "generator.keras")
    logger.info("Discriminator model: %s", MODELS_DIR / "discriminator.keras")
    logger.info("Training history: %s", history_file)
    logger.info("Loss graph: %s", PLOTS_DIR / "gan_training_loss.png")
    logger.info("=" * 60)

    return history_df


# ============================================================
# 11. COMMAND-LINE INTERFACE
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train a basic MNIST GAN."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Number of training epochs (default: {EPOCHS}).",
    )

    parser.add_argument(
        "--limit-samples",
        type=int,
        default=None,
        help="Optional number of MNIST training samples for a quick test.",
    )

    args = parser.parse_args()

    train_gan(
        epochs=args.epochs,
        limit_samples=args.limit_samples,
    )


if __name__ == "__main__":
    main()
