"""
decode_secret.py  --  Receiver-side decoder for Generative Steganography
=========================================================================
Usage:  python decode_secret.py

You will be prompted for:
  1. Path to the stego image (downloaded from Colab)
  2. Path to coeff_order.npy (downloaded from Colab)
  3. (Optional) The original secret message -- for accuracy testing
  4. (Optional) Path to a clean cover image -- for MSE / PSNR / SSIM

Configuration below MUST match the embedding notebook values.

Outputs saved alongside the stego image:
  decoder_loss_curve.png   -- per-step inversion loss L = a*L_cover + b*L_secret
  decoder_summary.png      -- full visual dashboard
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
import torch
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler
from PIL import Image

try:
    from skimage.metrics import structural_similarity as ssim_fn
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("[WARN] scikit-image not installed -- SSIM skipped.  pip install scikit-image")

# ?? Parameters: MUST match embedding notebook ?????????????????????????????????
MODEL_ID       = "runwayml/stable-diffusion-v1-5"
PROMPT         = "a scenic mountain landscape at sunset, highly detailed, oil painting"
NUM_STEPS      = 50
GUIDANCE_SCALE = 1.0
REPS           = 5
RING_R_LOW     = 6
RING_R_HIGH    = 28
EMBED_CHANNEL  = 0
SEED_KEY       = 1234
ALPHA          = 0.5   # loss weight for cover-fidelity term
BETA           = 0.5   # loss weight for secret-noise term

# ??????????????????????????????????????? helpers ??????????????????????????????
def text_to_bits(text):
    return np.unpackbits(np.frombuffer(text.encode("utf-8"), dtype=np.uint8))

def bits_to_text(bits):
    n = (len(bits) // 8) * 8
    return np.packbits(bits[:n].astype(np.uint8)).tobytes().decode("utf-8", errors="replace")

def ecc_decode(coded, n_orig, reps=REPS):
    coded = coded[: n_orig * reps].reshape(n_orig, reps)
    return (coded.mean(axis=1) >= 0.5).astype(np.uint8)

def get_ring_coords(h, w, r_low, r_high):
    fy = np.fft.fftfreq(h) * h
    fx = np.fft.fftfreq(w) * w
    ys, xs = np.meshgrid(fy, fx, indexing="ij")
    r = np.sqrt(ys**2 + xs**2)
    mask = (r >= r_low) & (r < r_high)
    coords = np.argwhere(mask)
    keep, seen = [], set()
    for y, x in coords:
        my, mx = (-y) % h, (-x) % w
        if (y, x) == (my, mx): continue
        if (my, mx) in seen:   continue
        seen.add((y, x)); keep.append((y, x))
    return np.array(keep)

def extract_bits_from_channel(channel, coords, order, n_bits):
    Z = np.fft.fft2(channel)
    bits = np.zeros(n_bits, dtype=np.uint8)
    for i, idx in enumerate(order):
        y, x = coords[idx]
        bits[i] = 1 if Z[y, x].real >= 0 else 0
    return bits

def plot_bits(ax, bits, title, max_show=120, highlight_reps=None):
    show = bits[:max_show]
    n = len(show); cols = min(n, 40); rows = int(np.ceil(n / cols))
    grid = np.full(rows * cols, -1, dtype=float)
    grid[:n] = show.astype(float)
    grid = grid.reshape(rows, cols)
    ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_title(title, fontsize=9, fontweight="bold"); ax.axis("off")
    if highlight_reps and highlight_reps > 1:
        for c in range(0, min(cols, n), highlight_reps):
            ax.axvline(c - 0.5, color="#4fc3f7", linewidth=0.8, alpha=0.7)

def mse_val(a, b):
    a = np.asarray(a, dtype=np.float64) / 255.0
    b = np.asarray(b, dtype=np.float64) / 255.0
    return float(np.mean((a - b) ** 2))

def psnr_val(mse, max_v=1.0):
    return float("inf") if mse == 0 else float(10 * np.log10(max_v**2 / mse))

def ssim_val(a, b):
    if not HAS_SKIMAGE: return None
    a = np.asarray(a, dtype=np.float64) / 255.0
    b = np.asarray(b, dtype=np.float64) / 255.0
    kw = dict(data_range=1.0)
    if a.ndim == 3: kw["channel_axis"] = -1
    return float(ssim_fn(a, b, **kw))

# ??????????????????????????????????????????????????????????????????????????????
def main():
    SEP = "=" * 65
    print(SEP)
    print("  GENERATIVE STEGANOGRAPHY -- RECEIVER / DECODER")
    print(SEP)

    # ?? 1. Collect paths ????????????????????????????????????????????????????
    print("\n[REQUIRED FILES]")
    stego_path = input("  Stego image path  [stego_image.png]  : ").strip() or "stego_image.png"
    coeff_path = input("  coeff_order.npy   [coeff_order.npy]  : ").strip() or "coeff_order.npy"

    if not os.path.exists(stego_path):
        sys.exit(f"\nERROR: '{stego_path}' not found.")
    if not os.path.exists(coeff_path):
        sys.exit(f"\nERROR: '{coeff_path}' not found.")

    # ?? 2. Optional reference message ???????????????????????????????????????
    print("\n[OPTIONAL -- for accuracy testing]")
    secret_ref = input("  Original secret message (blank to skip) : ").strip()
    have_msg   = bool(secret_ref)
    if have_msg:
        bits_ref       = text_to_bits(secret_ref)
        coded_bits_ref = np.repeat(bits_ref, REPS)
        print(f"  => {len(bits_ref)} raw bits / {len(coded_bits_ref)} ECC-coded bits")

    # ?? 3. Optional cover image ??????????????????????????????????????????????
    print("\n[OPTIONAL -- for MSE/PSNR/SSIM]")
    print("  Provide the clean cover image (same prompt+seed, no secret hidden).")
    print("  Generate it in the Colab notebook (see the dedicated cell).")
    cover_path = input("  Cover image path  (blank to skip) : ").strip()
    have_cover = bool(cover_path) and os.path.exists(cover_path)
    if cover_path and not have_cover:
        print(f"  [WARN] '{cover_path}' not found -- pixel metrics skipped.")

    out_dir = os.path.dirname(os.path.abspath(stego_path))

    # ?? 4. Load images ???????????????????????????????????????????????????????
    stego_img    = Image.open(stego_path).convert("RGB")
    stego_arr    = np.array(stego_img)
    coeff_order  = np.load(coeff_path)
    n_coded_bits = len(coeff_order)
    n_orig_bits  = n_coded_bits // REPS
    cover_img    = Image.open(cover_path).convert("RGB") if have_cover else None
    cover_arr    = np.array(cover_img) if have_cover else None

    print(f"\n  Stego : {stego_img.size}  |  {n_coded_bits} coded bits => {n_orig_bits} original bits")
    if have_cover: print(f"  Cover : {cover_img.size}")

    # ?? 5. Show received images ??????????????????????????????????????????????
    ncols = 2 if have_cover else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6*ncols, 6))
    if ncols == 1: axes = [axes]
    axes[0].imshow(stego_img)
    axes[0].set_title("DECODER Step 1\nStego image received", fontsize=11, fontweight="bold")
    axes[0].axis("off")
    if have_cover:
        axes[1].imshow(cover_img)
        axes[1].set_title("Cover image (no secret)\nFor pixel-domain metrics", fontsize=11, fontweight="bold")
        axes[1].axis("off")
    plt.tight_layout(); plt.show()

    # ?? 6. Load model ????????????????????????????????????????????????????????
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32
    print(f"\n  Device: {device} | dtype: {dtype}")
    print("  Loading frozen Stable Diffusion model ...")

    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype, safety_checker=None)
    pipe = pipe.to(device)
    inv_sched = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    for m in [pipe.unet, pipe.vae, pipe.text_encoder]:
        m.requires_grad_(False); m.eval()

    LATENT_SIZE = pipe.unet.config.sample_size   # 64
    VAE_SCALE   = pipe.vae.config.scaling_factor
    print(f"  Model ready.  Latent: 4 x {LATENT_SIZE} x {LATENT_SIZE}")

    ring_coords = get_ring_coords(LATENT_SIZE, LATENT_SIZE, RING_R_LOW, RING_R_HIGH)

    # ?? 7. VAE encode stego -> z0 ????????????????????????????????????????????
    print("\n  Step 2: VAE encode stego -> latent z0 ...")
    with torch.no_grad():
        img_t  = pipe.image_processor.preprocess(stego_img).to(device=device, dtype=dtype)
        z0_hat = pipe.vae.encode(img_t).latent_dist.mode() * VAE_SCALE

    z0_cover = None
    if have_cover:
        with torch.no_grad():
            cov_t    = pipe.image_processor.preprocess(cover_img).to(device=device, dtype=dtype)
            z0_cover = pipe.vae.encode(cov_t).latent_dist.mode() * VAE_SCALE

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    fig.suptitle("DECODER Step 2 -- VAE Encoded Latent z0 (4 channels)", fontsize=11, fontweight="bold")
    for ch in range(4):
        data = z0_hat[0, ch].float().cpu().numpy()
        im = axes[ch].imshow(data, cmap="RdBu_r", interpolation="nearest")
        axes[ch].set_title(f"Channel {ch}  sd={data.std():.2f}", fontsize=9)
        axes[ch].axis("off"); plt.colorbar(im, ax=axes[ch], shrink=0.85)
    plt.tight_layout(); plt.show()

    # ?? 8. DDIM inversion + per-step loss tracking ???????????????????????????
    print("  Step 3: DDIM inversion (image -> noise) + per-step loss ...")
    z0_ref = z0_cover if z0_cover is not None else z0_hat.clone()

    loss_cover_list  = []
    loss_secret_list = []
    loss_total_list  = []
    step_indices     = []
    snapshots        = {}
    snap_steps       = {0, NUM_STEPS//4, NUM_STEPS//2, 3*NUM_STEPS//4, NUM_STEPS-1}

    with torch.no_grad():
        prompt_embeds, _ = pipe.encode_prompt(
            PROMPT, device, num_images_per_prompt=1,
            do_classifier_free_guidance=False, negative_prompt=None)

        inv_sched.set_timesteps(NUM_STEPS, device=device)
        inv_latents = z0_hat.clone()

        for step_i, t in enumerate(inv_sched.timesteps):
            noise_pred  = pipe.unet(inv_latents, t, encoder_hidden_states=prompt_embeds).sample
            inv_latents = inv_sched.step(noise_pred, t, inv_latents).prev_sample

            if step_i in snap_steps:
                snapshots[step_i] = inv_latents[0, EMBED_CHANNEL].float().cpu().numpy().copy()

            # L_cover: latent drift from z0 reference
            l_cover = float(torch.mean((inv_latents.float() - z0_ref.float()) ** 2).item())

            # L_secret: embed-channel deviation from N(0,1) -- ideal recovered noise
            ch_flat  = inv_latents[0, EMBED_CHANNEL].float()
            l_secret = float((torch.mean(ch_flat)**2 + (ch_flat.std() - 1.0)**2).item())

            l_total = ALPHA * l_cover + BETA * l_secret
            loss_cover_list.append(l_cover)
            loss_secret_list.append(l_secret)
            loss_total_list.append(l_total)
            step_indices.append(step_i)

    print(f"  Inversion done.  Final total loss: {loss_total_list[-1]:.4f}")

    # ?? 9. Inversion progress plot ???????????????????????????????????????????
    snaps = sorted(snapshots.keys())
    fig, axes = plt.subplots(2, len(snaps), figsize=(15, 6))
    fig.suptitle("DECODER Step 3 -- DDIM Inversion Progress (Channel 0)", fontsize=11, fontweight="bold")
    for col, si in enumerate(snaps):
        data = snapshots[si]
        axes[0, col].imshow(data, cmap="RdBu_r", interpolation="nearest")
        axes[0, col].set_title(f"Step {si+1}/{NUM_STEPS}\nsd={data.std():.2f}", fontsize=9)
        axes[0, col].axis("off")
        Z = np.fft.fftshift(np.log1p(np.abs(np.fft.fft2(data))))
        axes[1, col].imshow(Z, cmap="inferno", interpolation="nearest")
        axes[1, col].set_title("Spectrum", fontsize=8); axes[1, col].axis("off")
    axes[0, 0].set_ylabel("Spatial", fontsize=9)
    axes[1, 0].set_ylabel("Spectrum", fontsize=9)
    plt.tight_layout(); plt.show()

    # ?? 10. Loss curves ??????????????????????????????????????????????????????
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        f"DECODER -- Inversion Loss  L = {ALPHA}*L_cover + {BETA}*L_secret",
        fontsize=12, fontweight="bold")

    axes[0].plot(step_indices, loss_cover_list, color="#4fc3f7", lw=1.8)
    axes[0].set_title("L_cover  (latent drift from z0)", fontsize=9)
    axes[0].set_xlabel("Inversion step"); axes[0].set_ylabel("MSE"); axes[0].grid(alpha=0.3)

    axes[1].plot(step_indices, loss_secret_list, color="#ff7043", lw=1.8)
    axes[1].set_title("L_secret  (embed-channel vs N(0,1))", fontsize=9)
    axes[1].set_xlabel("Inversion step"); axes[1].set_ylabel("Loss"); axes[1].grid(alpha=0.3)

    axes[2].plot(step_indices, loss_total_list, color="#a5d6a7", lw=2.2)
    axes[2].fill_between(step_indices, loss_total_list, alpha=0.15, color="#a5d6a7")
    axes[2].set_title(f"L_total = {ALPHA}*L_cover + {BETA}*L_secret", fontsize=9)
    axes[2].set_xlabel("Inversion step"); axes[2].set_ylabel("Loss"); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    loss_path = os.path.join(out_dir, "decoder_loss_curve.png")
    plt.savefig(loss_path, dpi=150, bbox_inches="tight"); plt.show()
    print(f"  Loss curve saved: {loss_path}")

    # ?? 11. Extract bits and decode message ??????????????????????????????????
    recovered_channel    = inv_latents[0, EMBED_CHANNEL].float().cpu().numpy()
    recovered_coded_bits = extract_bits_from_channel(
        recovered_channel, ring_coords, coeff_order, n_coded_bits)
    recovered_bits    = ecc_decode(recovered_coded_bits, n_orig_bits, REPS)
    recovered_message = bits_to_text(recovered_bits)

    # ?? 12. FFT ring read-out visualisation ??????????????????????????????????
    Z_rec = np.fft.fft2(recovered_channel)
    bit_map_rec = np.full((LATENT_SIZE, LATENT_SIZE), np.nan)
    for i, idx in enumerate(coeff_order[:n_coded_bits]):
        y, x = ring_coords[idx]
        sy = (y + LATENT_SIZE // 2) % LATENT_SIZE
        sx = (x + LATENT_SIZE // 2) % LATENT_SIZE
        bit_map_rec[sy, sx] = recovered_coded_bits[i]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("DECODER Step 4 -- FFT of Recovered Noise -- Reading Bits from Ring",
                 fontsize=11, fontweight="bold")
    base = np.log1p(np.abs(np.fft.fftshift(Z_rec)))
    axes[0].imshow(base, cmap="Greys_r", interpolation="nearest")
    sc = axes[0].imshow(bit_map_rec, cmap="RdYlGn", vmin=0, vmax=1,
                        interpolation="nearest", alpha=0.9)
    axes[0].set_title("Bits on spectrum (green=1, red=0)", fontsize=9)
    plt.colorbar(sc, ax=axes[0], shrink=0.8, label="bit value")
    cx = cy = LATENT_SIZE // 2
    for rad, ls in [(RING_R_LOW, "--"), (RING_R_HIGH, "-")]:
        axes[0].add_patch(Circle((cx, cy), rad, fill=False, edgecolor="#4fc3f7", lw=1.2, ls=ls))
    real_parts = [Z_rec[ring_coords[idx][0], ring_coords[idx][1]].real
                  for idx in coeff_order[:n_coded_bits]]
    axes[1].bar(range(len(real_parts)), real_parts,
                color=["#4caf50" if v > 0 else "#f44336" for v in real_parts],
                alpha=0.8, width=1.0)
    axes[1].axhline(0, color="white", lw=0.8)
    axes[1].set_xlabel("Bit position", fontsize=9); axes[1].set_ylabel("Re(Z[y,x])", fontsize=9)
    axes[1].set_title("Sign -> bit:  + -> 1,  - -> 0", fontsize=9, fontweight="bold")
    axes[1].grid(alpha=0.2)
    plot_bits(axes[2], recovered_coded_bits, f"Extracted coded bits ({n_coded_bits})")
    plt.tight_layout(); plt.show()

    # ?? 13. ECC decode visualisation ?????????????????????????????????????????
    fig, axes = plt.subplots(1, 2, figsize=(12, 3))
    fig.suptitle("DECODER Step 5 -- ECC Majority-Vote -> Final Bits", fontsize=11, fontweight="bold")
    plot_bits(axes[0], recovered_coded_bits,
              f"Coded bits (x{REPS} redundancy, {n_coded_bits} total)", highlight_reps=REPS)
    plot_bits(axes[1], recovered_bits, f"Decoded bits ({n_orig_bits}) after majority vote")
    plt.tight_layout(); plt.show()

    # ?? 14. Bit-error analysis ???????????????????????????????????????????????
    ber_raw = ber_ecc = match = char_acc = bit_acc = None
    if have_msg:
        mc = min(len(coded_bits_ref), len(recovered_coded_bits))
        mb = min(len(bits_ref),       len(recovered_bits))
        ber_raw  = float(np.mean(recovered_coded_bits[:mc] != coded_bits_ref[:mc]))
        ber_ecc  = float(np.mean(recovered_bits[:mb]        != bits_ref[:mb]))
        match    = recovered_message.strip() == secret_ref.strip()
        char_acc = sum(a == b for a, b in zip(recovered_message[:len(secret_ref)], secret_ref)) \
                   / max(len(secret_ref), 1) * 100
        bit_acc  = (1.0 - ber_raw) * 100

        fig, axes = plt.subplots(1, 3, figsize=(15, 3))
        fig.suptitle("DECODER Step 6 -- Bit Error Analysis (vs. known secret)",
                     fontsize=11, fontweight="bold")
        plot_bits(axes[0], coded_bits_ref,       f"ORIGINAL coded bits ({len(coded_bits_ref)})")
        plot_bits(axes[1], recovered_coded_bits, f"RECOVERED coded  BER={ber_raw*100:.1f}%")
        errors = (recovered_coded_bits[:mc] != coded_bits_ref[:mc]).astype(float)
        nc = len(errors); cols = min(nc, 40); rows = int(np.ceil(nc / cols))
        g = np.full(rows * cols, np.nan); g[:nc] = errors; g = g.reshape(rows, cols)
        axes[2].imshow(g, cmap="RdYlGn_r", vmin=0, vmax=1, interpolation="nearest", aspect="auto")
        axes[2].set_title(f"Flip positions (red=error)  {int(errors.sum())}/{nc}", fontsize=9)
        axes[2].axis("off")
        plt.tight_layout(); plt.show()

    # ?? 15. Pixel-domain evaluation ??????????????????????????????????????????
    pix = {}
    if have_cover:
        if stego_arr.shape != cover_arr.shape:
            cover_arr = np.array(cover_img.resize(stego_img.size, Image.LANCZOS))
        mse  = mse_val(stego_arr, cover_arr)
        psnr = psnr_val(mse)
        ssim = ssim_val(stego_arr, cover_arr)
        pix  = dict(mse=mse, psnr=psnr, ssim=ssim)
        diff = np.abs(stego_arr.astype(np.int16) - cover_arr.astype(np.int16)).astype(np.uint8)

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        fig.suptitle("DECODER Step 7 -- Pixel-Domain Image Quality Evaluation",
                     fontsize=11, fontweight="bold")
        axes[0].imshow(cover_arr); axes[0].axis("off")
        axes[0].set_title("Cover image (no secret)", fontsize=9, fontweight="bold")
        axes[1].imshow(stego_arr); axes[1].axis("off")
        axes[1].set_title("Stego image (secret embedded)", fontsize=9, fontweight="bold")
        im = axes[2].imshow(diff, cmap="hot", interpolation="nearest"); axes[2].axis("off")
        ss = f"{ssim:.4f}" if ssim is not None else "N/A"
        axes[2].set_title(f"Pixel difference\nMSE={mse:.4f}  PSNR={psnr:.2f}dB  SSIM={ss}",
                          fontsize=9, fontweight="bold")
        plt.colorbar(im, ax=axes[2], shrink=0.85)
        plt.tight_layout(); plt.show()

    # ?? 16. Full summary dashboard ???????????????????????????????????????????
    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

    def mini(ax, b, title):
        n = len(b[:80]); cols = min(n, 40); rows = int(np.ceil(n / cols))
        g = np.full(rows * cols, -1.0)
        g[:n] = b[:n].astype(float); g = g.reshape(rows, cols)
        ax.imshow(g, cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest", aspect="auto")
        ax.set_title(title, fontsize=8, fontweight="bold"); ax.axis("off")

    ax00 = fig.add_subplot(gs[0, 0]); ax00.imshow(stego_img); ax00.axis("off")
    ax00.set_title("1. Stego image\n(received)", fontsize=8, fontweight="bold")

    ax01 = fig.add_subplot(gs[0, 1])
    ax01.imshow(recovered_channel, cmap="RdBu_r", interpolation="nearest"); ax01.axis("off")
    ax01.set_title("2. Recovered noise\n(Ch 0, DDIM inv)", fontsize=8, fontweight="bold")

    ax02 = fig.add_subplot(gs[0, 2])
    base2 = np.log1p(np.abs(np.fft.fftshift(Z_rec)))
    ax02.imshow(base2, cmap="Greys_r", interpolation="nearest")
    ax02.imshow(bit_map_rec, cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest", alpha=0.85)
    ax02.set_title("3. Bits in\nfrequency ring", fontsize=8, fontweight="bold"); ax02.axis("off")

    ax03 = fig.add_subplot(gs[0, 3])
    ax03.plot(step_indices, loss_total_list, color="#a5d6a7", lw=1.8)
    ax03.fill_between(step_indices, loss_total_list, alpha=0.15, color="#a5d6a7")
    ax03.set_title(f"4. Inversion loss\nL={ALPHA}*Lcov+{BETA}*Lsec", fontsize=8, fontweight="bold")
    ax03.set_xlabel("Step", fontsize=8); ax03.grid(alpha=0.25)

    ax10 = fig.add_subplot(gs[1, 0]); mini(ax10, recovered_coded_bits, f"5. Extracted coded bits\n({n_coded_bits})")
    ax11 = fig.add_subplot(gs[1, 1]); mini(ax11, recovered_bits, f"6. Decoded bits\n({n_orig_bits})")

    ax12 = fig.add_subplot(gs[1, 2]); ax12.axis("off")
    rl = ["RECOVERED MESSAGE", "-"*28, f"  {recovered_message!r}", ""]
    if have_msg:
        rl += ["SECRET RECOVERY ACCURACY",
               f"  Char accuracy  : {char_acc:.1f}%",
               f"  Bit acc (coded): {bit_acc:.1f}%",
               f"  BER raw        : {ber_raw*100:.2f}%",
               f"  BER after ECC  : {ber_ecc*100:.2f}%",
               f"  Exact match    : {'YES' if match else 'NO'}"]
    else:
        rl.append("(no reference -- accuracy skipped)")
    ax12.text(0.05, 0.5, "\n".join(rl), fontsize=9, family="monospace", va="center",
              transform=ax12.transAxes,
              bbox=dict(boxstyle="round", facecolor="#1e3a5f", edgecolor="#4fc3f7", alpha=0.9),
              color="white")
    ax12.set_title("Decoding Result", fontsize=9, fontweight="bold")

    ax13 = fig.add_subplot(gs[1, 3]); ax13.axis("off")
    el = ["IMAGE QUALITY METRICS", "-"*28]
    if pix:
        ss = f"{pix['ssim']:.4f}" if pix.get("ssim") is not None else "N/A"
        el += [f"  MSE  : {pix['mse']:.6f}",
               f"  PSNR : {pix['psnr']:.2f} dB",
               f"  SSIM : {ss}", "",
               f"  Final loss : {loss_total_list[-1]:.4f}",
               f"  alpha={ALPHA}, beta={BETA}"]
    else:
        el += ["  (no cover image -- skipped)", "",
               f"  Final L_total: {loss_total_list[-1]:.4f}",
               f"  alpha={ALPHA}, beta={BETA}"]
    ax13.text(0.05, 0.5, "\n".join(el), fontsize=9, family="monospace", va="center",
              transform=ax13.transAxes,
              bbox=dict(boxstyle="round", facecolor="#1a3a1a", edgecolor="#4caf50", alpha=0.9),
              color="#e8f5e9")
    ax13.set_title("Evaluation Metrics", fontsize=9, fontweight="bold")

    fig.suptitle("Generative Steganography -- Receiver Decoder Summary Dashboard",
                 fontsize=13, fontweight="bold")
    summary_path = os.path.join(out_dir, "decoder_summary.png")
    plt.savefig(summary_path, dpi=150, bbox_inches="tight"); plt.show()

    # ?? 17. Terminal summary ?????????????????????????????????????????????????
    print(); print(SEP)
    print(f"  RECOVERED SECRET : {recovered_message!r}")
    if have_msg:
        print(f"  Reference        : {secret_ref!r}")
        print(f"  Exact match      : {'YES' if match else 'NO'}")
        print(f"  Char accuracy    : {char_acc:.1f}%")
        print(f"  Bit acc (coded)  : {bit_acc:.1f}%")
        print(f"  BER raw / ECC    : {ber_raw*100:.2f}% / {ber_ecc*100:.2f}%")
    if pix:
        ss = f"{pix['ssim']:.4f}" if pix.get("ssim") is not None else "N/A"
        print(f"  MSE              : {pix['mse']:.6f}")
        print(f"  PSNR             : {pix['psnr']:.2f} dB")
        print(f"  SSIM             : {ss}")
    print(f"  Final loss       : {loss_total_list[-1]:.4f}  "
          f"(L = {ALPHA}*L_cover + {BETA}*L_secret)")
    print(f"  Summary saved    : {summary_path}")
    print(f"  Loss curve saved : {loss_path}")
    print(SEP)

if __name__ == "__main__":
    main()
