import json

# ─────────────────────────────────────────────────────────────────────────────
# NOTEBOOK 1 — Jupyter: Enhanced Embedding with Step-by-Step Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def C(src):   # code cell
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src}

def M(src):   # markdown cell
    return {"cell_type":"markdown","metadata":{},"source":src}

embed_cells = [

M("""# 🔐 Generative Steganography — Full Visual Demonstration
### Stable Diffusion v1-5 · DDIM · Fourier-Ring Embedding · No Training Required

**What you will see, step by step:**
1. Your secret message converted to binary bits
2. ECC-encoded (repetition-coded) bit stream
3. Pure Gaussian noise (baseline) vs structured noise (after bit embedding)
4. Fourier-spectrum heatmap showing *exactly where* the bits are hidden
5. The AI-generated stego image
6. DDIM inversion recovering the noise
7. Bit extraction and ECC decoding → original message
"""),

# ── Section 0: install / imports ──────────────────────────────────────────────
M("## 0 · Install & Imports"),

C("""\
# Uncomment once if needed:
# !pip install -q diffusers transformers accelerate torch matplotlib numpy

import torch, numpy as np, matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDIMInverseScheduler
from matplotlib.patches import Circle, FancyArrowPatch
import warnings; warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if device == "cuda" else torch.float32
print(f"✓ Running on  : {device}")
print(f"✓ Tensor dtype: {dtype}")
"""),

# ── Section 1: load model ─────────────────────────────────────────────────────
M("""## 1 · Load Frozen Stable-Diffusion (no training, no fine-tuning)

We download `runwayml/stable-diffusion-v1-5` **once** and freeze every weight.  
The only thing we customise is the *initial noise latent* — the U-Net and VAE are  
used exactly as published.

**Architecture reminder:**
```
User prompt ──► Text Encoder ──► Condition embeddings
                                          │
Random noise ──► U-Net (50 DDIM steps) ──► Denoised latent ──► VAE Decoder ──► 512×512 image
```
"""),

C("""\
MODEL_ID = "runwayml/stable-diffusion-v1-5"
print("Loading model — this may download ~4 GB the first time …")

pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype, safety_checker=None)
pipe = pipe.to(device)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)

# Freeze ALL parameters
for m in [pipe.unet, pipe.vae, pipe.text_encoder]:
    m.requires_grad_(False); m.eval()

LATENT_CHANNELS = pipe.unet.config.in_channels   # 4
LATENT_SIZE     = pipe.unet.config.sample_size   # 64
VAE_SCALE       = pipe.vae.config.scaling_factor

print(f"✓ Model loaded & frozen")
print(f"  Latent shape  : (1, {LATENT_CHANNELS}, {LATENT_SIZE}, {LATENT_SIZE})")
print(f"  Pixel image   : 512 × 512")
print(f"  VAE scale     : {VAE_SCALE}")
"""),

# ── Section 2: ECC helpers ────────────────────────────────────────────────────
M("""## 2 · Message ↔ Bits with Repetition ECC

A **repetition code** repeats every bit `REPS` times.  
On recovery, majority vote corrects isolated flips caused by the VAE → DDIM  
round-trip imprecision.
"""),

C("""\
REPS = 5   # increase to 7-9 if you still see errors after tuning

def text_to_bits(text):
    b = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
    return np.unpackbits(b)

def bits_to_text(bits):
    n = (len(bits) // 8) * 8
    b = np.packbits(bits[:n].astype(np.uint8))
    return b.tobytes().decode("utf-8", errors="replace")

def ecc_encode(bits, reps=REPS):
    return np.repeat(bits, reps)

def ecc_decode(coded, n_orig, reps=REPS):
    coded = coded[: n_orig * reps].reshape(n_orig, reps)
    return (coded.mean(axis=1) >= 0.5).astype(np.uint8)

# ── Visual helper: plot a bit-stream as coloured squares ──────────────────────
def plot_bits(ax, bits, title, max_show=120, highlight_reps=None):
    show = bits[:max_show]
    n = len(show)
    cols = min(n, 40)
    rows = int(np.ceil(n / cols))
    grid = np.full(rows * cols, -1, dtype=float)
    grid[:n] = show.astype(float)
    grid = grid.reshape(rows, cols)

    cmap = mcolors.ListedColormap(["#2d2d2d", "#f4f4f4", "#1a1a1a"])
    ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto",
              interpolation="nearest")
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.axis("off")
    if highlight_reps and highlight_reps > 1:
        for col in range(0, min(cols, n), highlight_reps):
            ax.axvline(col - 0.5, color="#4fc3f7", linewidth=0.8, alpha=0.7)
"""),

# ── Section 3: Fourier helpers ────────────────────────────────────────────────
M("""## 3 · Fourier-Ring Embedding Algorithm

**Why Fourier domain?**  
The latent space of Stable Diffusion is spatially structured — high-frequency  
noise is suppressed during generation.  By placing bits in a *mid-frequency ring*  
we ensure they survive the U-Net's denoising pass and can be read back via inversion.

**Embedding rule (per bit):**
```
FFT(channel)  →  find ring coefficient Z[y,x]
  bit = 1  ⟹  force Re(Z[y,x])  to  +|Z[y,x]| + strength
  bit = 0  ⟹  force Re(Z[y,x])  to  −|Z[y,x]| − strength
  (mirror Hermitian conjugate so IFFT remains real)
IFFT  →  modified channel (still Gaussian-looking)
```
"""),

C("""\
RING_R_LOW, RING_R_HIGH = 6, 28
EMBED_STRENGTH = 10.0
EMBED_CHANNEL  = 0
SEED_KEY       = 1234

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
        if (my, mx) in seen:    continue
        seen.add((y, x)); keep.append((y, x))
    return np.array(keep)

def embed_bits_into_channel(channel, bits, coords, strength, key):
    h, w = channel.shape
    assert len(bits) <= len(coords), "Message too long — raise RING_R_HIGH or shorten message"
    Z = np.fft.fft2(channel)
    order = np.random.default_rng(key).permutation(len(coords))[:len(bits)]
    for bit, idx in zip(bits, order):
        y, x   = coords[idx]
        my, mx = (-y) % h, (-x) % w
        mag    = np.abs(Z[y, x]) + strength
        sign   = 1.0 if bit == 1 else -1.0
        nv     = complex(sign * mag, Z[y, x].imag)
        Z[y, x] = nv; Z[my, mx] = np.conj(nv)
    return np.fft.ifft2(Z).real, order

def extract_bits_from_channel(channel, coords, order, n_bits):
    Z = np.fft.fft2(channel)
    bits = np.zeros(n_bits, dtype=np.uint8)
    for i, idx in enumerate(order):
        y, x = coords[idx]
        bits[i] = 1 if Z[y, x].real >= 0 else 0
    return bits

print("✓ Fourier embedding functions defined")
"""),

# ── Section 4: user inputs ────────────────────────────────────────────────────
M("## 4 · Enter Your Secret Message & Prompt"),

C("""\
secret_message = "MEET AT DAWN"   # ← change me
prompt         = "a scenic mountain landscape at sunset, highly detailed, oil painting"
NUM_STEPS      = 50
GUIDANCE_SCALE = 1.0   # keep at 1.0 for accurate DDIM inversion

bits       = text_to_bits(secret_message)
coded_bits = ecc_encode(bits, REPS)

print("━"*60)
print(f" Secret message : {secret_message!r}")
print(f" UTF-8 bytes    : {list(secret_message.encode())}")
print(f" Raw bits       : {len(bits)}  bits  ({len(bits)//8} bytes)")
print(f" After x{REPS} ECC  : {len(coded_bits)} bits")
print("━"*60)
"""),

# ── Section 5: VISUALISE BIT CONVERSION ──────────────────────────────────────
M("""## 5 · Visualise: Message → Bits → ECC Bits

Here you can see **exactly** what your text looks like as binary,  
and how the repetition ECC expands each bit into `REPS` copies.
"""),

C("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 3))
fig.suptitle("Step 1 · Text → Binary → ECC-encoded Bit Stream", fontsize=12, fontweight="bold")

plot_bits(axes[0], bits,       f"Raw bits  ({len(bits)} total) — one square = one bit")
plot_bits(axes[1], coded_bits, f"ECC bits  ({len(coded_bits)} total, x{REPS} repetition) — vertical lines show groups", highlight_reps=REPS)

# Annotate individual characters
char_boundaries = [i*8 for i in range(len(secret_message)+1)]
ax = axes[0]
for i, ch in enumerate(secret_message):
    col = (i * 8) % 40
    row = (i * 8) // 40
    ax.text(col + 3.5, row, ch, ha="center", va="center",
            fontsize=7, color="#ff6b35", fontweight="bold")

plt.tight_layout()
plt.show()

# Also print the actual bit values per character
print("\\nBit breakdown per character:")
print(f"  {'Char':>5} | {'ASCII':>5} | {'Binary':>10}")
print("  " + "-"*30)
for ch in secret_message:
    code = ord(ch)
    bstr = format(code, "08b")
    print(f"  {ch!r:>5} | {code:>5} | {bstr:>10}")
"""),

# ── Section 6: build raw noise ────────────────────────────────────────────────
M("""## 6 · Generate Base Gaussian Noise

This is the **raw starting noise** for Stable Diffusion — completely random, no secret yet.  
Shape: `(1, 4, 64, 64)` → 4 channels × 64×64 latent grid.
"""),

C("""\
generator = torch.Generator(device=device).manual_seed(SEED_KEY)
raw_latents = torch.randn(
    (1, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE),
    generator=generator, device=device, dtype=torch.float32
)

fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
fig.suptitle("Step 2 · Raw Gaussian Noise (4 Latent Channels) — No Secret Yet",
             fontsize=12, fontweight="bold")

for ch in range(4):
    data = raw_latents[0, ch].cpu().numpy()
    im = axes[ch].imshow(data, cmap="RdBu_r", interpolation="nearest")
    axes[ch].set_title(f"Channel {ch}  μ={data.mean():.2f}  σ={data.std():.2f}", fontsize=9)
    axes[ch].axis("off")
    plt.colorbar(im, ax=axes[ch], shrink=0.85)

plt.tight_layout()
plt.show()

print(f"Noise shape : {tuple(raw_latents.shape)}")
print(f"Global mean : {raw_latents.mean():.4f}   (should be ≈ 0)")
print(f"Global std  : {raw_latents.std():.4f}    (should be ≈ 1)")
"""),

# ── Section 7: Fourier spectrum BEFORE embedding ──────────────────────────────
M("""## 7 · Fourier Spectrum of Raw Noise (Before Embedding)

A **FFT magnitude heatmap** of Channel 0.  
The ring region (r_low to r_high) is highlighted — this is where the bits will go.  
Before embedding: it is just uniform random noise.
"""),

C("""\
ring_coords = get_ring_coords(LATENT_SIZE, LATENT_SIZE, RING_R_LOW, RING_R_HIGH)
print(f"Ring coefficients available : {len(ring_coords)}  (need ≥ {len(coded_bits)})")

ch0_raw = raw_latents[0, EMBED_CHANNEL].cpu().numpy()
Z_raw   = np.fft.fft2(ch0_raw)
Z_mag   = np.fft.fftshift(np.log1p(np.abs(Z_raw)))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Step 3 · FFT Spectrum of Channel 0 — BEFORE Bit Embedding", fontsize=12, fontweight="bold")

# Spectrum heatmap
im = axes[0].imshow(Z_mag, cmap="inferno", interpolation="nearest")
axes[0].set_title("Log-magnitude spectrum (shifted)", fontsize=9)
plt.colorbar(im, ax=axes[0], shrink=0.85)

# Ring overlay
cx, cy = LATENT_SIZE // 2, LATENT_SIZE // 2
for r, ls in [(RING_R_LOW, "--"), (RING_R_HIGH, "-")]:
    circ = Circle((cx, cy), r, fill=False, edgecolor="#00e5ff", linewidth=1.5, linestyle=ls)
    axes[0].add_patch(circ)
axes[0].text(cx + RING_R_HIGH + 1, cy, "← embedding ring", color="#00e5ff", fontsize=7, va="center")

# Ring coordinate scatter
ry = [c[0] for c in ring_coords]
rx = [c[1] for c in ring_coords]
axes[1].scatter(rx, ry, s=6, c="#00e5ff", alpha=0.6, label="ring positions")
axes[1].set_xlim(0, LATENT_SIZE); axes[1].set_ylim(LATENT_SIZE, 0)
axes[1].set_title(f"Ring coefficient positions ({len(ring_coords)} points)", fontsize=9)
axes[1].set_xlabel("Frequency X"); axes[1].set_ylabel("Frequency Y")
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.2)

plt.tight_layout(); plt.show()
"""),

# ── Section 8: embed bits ─────────────────────────────────────────────────────
M("""## 8 · Embed Bits into Latent Noise

The algorithm modifies Channel 0 in Fourier space:
- **bit = 1** → real part of ring coefficient forced **positive**
- **bit = 0** → real part of ring coefficient forced **negative**

The resulting latent still looks like Gaussian noise — the changes are sub-perceptual.
"""),

C("""\
latents = raw_latents.clone()
ch0_arr = latents[0, EMBED_CHANNEL].cpu().numpy()

embedded_ch, coeff_order = embed_bits_into_channel(
    ch0_arr, coded_bits, ring_coords, EMBED_STRENGTH, key=SEED_KEY
)
latents[0, EMBED_CHANNEL] = torch.from_numpy(embedded_ch).to(device=device, dtype=torch.float32)

# ── Compare raw vs embedded channel ──────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Step 4 · Noise Before vs After Secret Bit Embedding", fontsize=12, fontweight="bold")

# Spatial maps
vmin = min(ch0_arr.min(), embedded_ch.min())
vmax = max(ch0_arr.max(), embedded_ch.max())

axes[0,0].imshow(ch0_arr, cmap="RdBu_r", vmin=vmin, vmax=vmax, interpolation="nearest")
axes[0,0].set_title("Channel 0 — RAW noise", fontsize=9); axes[0,0].axis("off")

axes[0,1].imshow(embedded_ch, cmap="RdBu_r", vmin=vmin, vmax=vmax, interpolation="nearest")
axes[0,1].set_title("Channel 0 — AFTER embedding", fontsize=9); axes[0,1].axis("off")

diff = embedded_ch - ch0_arr
im = axes[0,2].imshow(diff, cmap="PiYG", interpolation="nearest")
axes[0,2].set_title(f"Difference (max Δ = {np.abs(diff).max():.3f})", fontsize=9)
axes[0,2].axis("off"); plt.colorbar(im, ax=axes[0,2], shrink=0.8)

# Fourier spectra comparison
Z_emb = np.fft.fft2(embedded_ch)
Z_emb_mag = np.fft.fftshift(np.log1p(np.abs(Z_emb)))

axes[1,0].imshow(Z_mag, cmap="inferno", interpolation="nearest")
axes[1,0].set_title("Spectrum — RAW", fontsize=9); axes[1,0].axis("off")

axes[1,1].imshow(Z_emb_mag, cmap="inferno", interpolation="nearest")
axes[1,1].set_title("Spectrum — AFTER embedding", fontsize=9); axes[1,1].axis("off")

# Embedded bit positions highlighted in spectrum
spec_diff = np.abs(np.fft.fftshift(Z_emb) - np.fft.fftshift(Z_raw))
axes[1,2].imshow(spec_diff, cmap="hot", interpolation="nearest")
axes[1,2].set_title("Spectrum Δ — modified coefficients (bright = changed)", fontsize=9)
axes[1,2].axis("off")

plt.tight_layout(); plt.show()

# Stats
print(f"Max spatial diff : {np.abs(diff).max():.4f}")
print(f"Noise std before : {ch0_arr.std():.4f}")
print(f"Noise std after  : {embedded_ch.std():.4f}")
"""),

# ── Section 9: show embedded bits in ring ─────────────────────────────────────
M("## 9 · Zoom In — Bit Pattern in Ring Coefficients"),

C("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Step 5 · Where Each Bit Lives in the Frequency Ring", fontsize=12, fontweight="bold")

Z_shift = np.fft.fftshift(Z_emb)
cx, cy  = LATENT_SIZE // 2, LATENT_SIZE // 2

# Colour each ring position by its encoded bit
bit_map = np.full((LATENT_SIZE, LATENT_SIZE), np.nan)
for i, idx in enumerate(coeff_order):
    y, x = ring_coords[idx]
    sy = (y + LATENT_SIZE // 2) % LATENT_SIZE
    sx = (x + LATENT_SIZE // 2) % LATENT_SIZE
    bit_map[sy, sx] = coded_bits[i]

base = np.log1p(np.abs(Z_shift))
axes[0].imshow(base, cmap="Greys_r", interpolation="nearest")
sc = axes[0].imshow(bit_map, cmap="RdYlGn", vmin=0, vmax=1,
                    interpolation="nearest", alpha=0.9)
axes[0].set_title("Embedded bits overlaid on spectrum\n(green=1, red=0, grey=unused)", fontsize=9)
plt.colorbar(sc, ax=axes[0], shrink=0.8, label="bit value")
for r, ls in [(RING_R_LOW,"--"),(RING_R_HIGH,"-")]:
    axes[0].add_patch(Circle((cx,cy), r, fill=False, edgecolor="#4fc3f7", linewidth=1.2, linestyle=ls))

# Real-part sign of embedded coefficients
real_parts = []
for idx in coeff_order:
    y, x = ring_coords[idx]
    real_parts.append(Z_emb[y, x].real)

axes[1].bar(range(len(real_parts)), real_parts,
            color=["#4caf50" if v > 0 else "#f44336" for v in real_parts],
            alpha=0.8, width=1.0)
axes[1].axhline(0, color="white", linewidth=0.8)
axes[1].set_xlabel("Coefficient index (bit position)", fontsize=9)
axes[1].set_ylabel("Real part of Z[y,x]", fontsize=9)
axes[1].set_title("Sign encodes bit: +ve → bit 1, -ve → bit 0", fontsize=9, fontweight="bold")
axes[1].grid(alpha=0.2)

plt.tight_layout(); plt.show()
print(f"First 16 embedded bits : {coded_bits[:16]}")
print(f"Corresponding signs    : {['+ ' if v>0 else '- ' for v in real_parts[:16]]}")
"""),

# ── Section 10: generate the image ───────────────────────────────────────────
M("""## 10 · Generate the Stego Image

The **frozen** U-Net runs 50 DDIM steps on the structured noise.  
Output: a perfectly normal-looking AI image that secretly contains your message.

> **Note:** `GUIDANCE_SCALE = 1.0` (CFG off) is required so DDIM inversion  
> faithfully retraces the same trajectory in reverse.
"""),

C("""\
latents_f16 = latents.to(dtype)

print("Running Stable Diffusion (this may take a minute on CPU) …")
with torch.no_grad():
    result = pipe(
        prompt=prompt,
        latents=latents_f16,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        output_type="pil",
    )
stego_image = result.images[0]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Step 6 · Generated Stego Image — Secret is Hidden Inside",
             fontsize=12, fontweight="bold")

axes[0].imshow(stego_image)
axes[0].set_title(f"Stego image\\nPrompt: \\\"{prompt[:55]}…\\\"", fontsize=9)
axes[0].axis("off")

# Show the structured noise used to generate it (embed channel)
axes[1].imshow(embedded_ch, cmap="RdBu_r", interpolation="nearest")
axes[1].set_title("The structured noise that produced this image\\n(Channel 0 — secret embedded here)", fontsize=9)
axes[1].axis("off")

plt.tight_layout(); plt.show()

# Save stego image (alongside coeff_order for reveal step)
stego_image.save("stego_image.png")
np.save("coeff_order.npy", coeff_order)
np.save("coded_bits_original.npy", coded_bits)
np.save("bits_original.npy", bits)
print("✓ Saved: stego_image.png  |  coeff_order.npy  |  coded_bits_original.npy  |  bits_original.npy")
"""),

# ── Section 11: DDIM inversion ────────────────────────────────────────────────
M("""## 11 · DDIM Inversion — Tracing the Image Back to Noise

DDIM inversion runs the **same frozen U-Net backwards** through all 50 timesteps,  
converting the stego pixel image back to an approximation of the original noise latent.

```
Stego image  →  VAE encoder  →  z₀  →  DDIM inverse (t=0→T)  →  z_T  ≈  original noise
```
"""),

C("""\
print("Running VAE encode + DDIM inversion …")
inv_snapshots = {}   # capture intermediate latents for visualisation

with torch.no_grad():
    img_t = pipe.image_processor.preprocess(stego_image).to(device=device, dtype=dtype)
    z0_hat = pipe.vae.encode(img_t).latent_dist.mode() * VAE_SCALE

    prompt_embeds, _ = pipe.encode_prompt(
        prompt, device, num_images_per_prompt=1,
        do_classifier_free_guidance=False, negative_prompt=None,
    )

    inverse_scheduler.set_timesteps(NUM_STEPS, device=device)
    inv_latents = z0_hat

    snap_steps = {0, NUM_STEPS//4, NUM_STEPS//2, 3*NUM_STEPS//4, NUM_STEPS-1}
    for step_i, t in enumerate(inverse_scheduler.timesteps):
        noise_pred   = pipe.unet(inv_latents, t, encoder_hidden_states=prompt_embeds).sample
        inv_latents  = inverse_scheduler.step(noise_pred, t, inv_latents).prev_sample
        if step_i in snap_steps:
            inv_snapshots[step_i] = inv_latents[0, EMBED_CHANNEL].float().cpu().numpy().copy()

print("✓ DDIM inversion complete")
"""),

M("## 12 · Visualise Inversion Progress"),

C("""\
snaps = sorted(inv_snapshots.keys())
fig, axes = plt.subplots(2, len(snaps), figsize=(15, 6))
fig.suptitle("Step 7 · DDIM Inversion Progress — Image → Noise (Channel 0)",
             fontsize=12, fontweight="bold")

for col, step_i in enumerate(snaps):
    data = inv_snapshots[step_i]
    axes[0, col].imshow(data, cmap="RdBu_r", interpolation="nearest")
    axes[0, col].set_title(f"Step {step_i+1}/{NUM_STEPS}\\nσ={data.std():.2f}", fontsize=9)
    axes[0, col].axis("off")

    Z = np.fft.fftshift(np.log1p(np.abs(np.fft.fft2(data))))
    axes[1, col].imshow(Z, cmap="inferno", interpolation="nearest")
    axes[1, col].set_title("Spectrum", fontsize=8)
    axes[1, col].axis("off")

# Label rows
axes[0, 0].set_ylabel("Spatial", fontsize=9, labelpad=4)
axes[1, 0].set_ylabel("Spectrum", fontsize=9, labelpad=4)

plt.tight_layout(); plt.show()

# Compare recovered noise to original
recovered_channel = inv_latents[0, EMBED_CHANNEL].float().cpu().numpy()
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle("Step 8 · Original Noise vs Recovered Noise (Channel 0)",
             fontsize=12, fontweight="bold")

axes[0].imshow(embedded_ch, cmap="RdBu_r", interpolation="nearest")
axes[0].set_title("Original embedded noise\\n(what we started with)", fontsize=9); axes[0].axis("off")

axes[1].imshow(recovered_channel, cmap="RdBu_r", interpolation="nearest")
axes[1].set_title("Recovered noise\\n(via DDIM inversion)", fontsize=9); axes[1].axis("off")

diff2 = np.abs(embedded_ch - recovered_channel)
im = axes[2].imshow(diff2, cmap="hot", interpolation="nearest")
axes[2].set_title(f"Absolute difference\\nmean Δ={diff2.mean():.4f}", fontsize=9); axes[2].axis("off")
plt.colorbar(im, ax=axes[2], shrink=0.8)

plt.tight_layout(); plt.show()
print(f"Noise recovery correlation: {np.corrcoef(embedded_ch.ravel(), recovered_channel.ravel())[0,1]:.4f}")
"""),

# ── Section 13: decode bits ────────────────────────────────────────────────────
M("## 13 · Extract Bits from Recovered Noise"),

C("""\
recovered_coded_bits = extract_bits_from_channel(
    recovered_channel, ring_coords, coeff_order, len(coded_bits)
)
recovered_bits    = ecc_decode(recovered_coded_bits, len(bits), REPS)
recovered_message = bits_to_text(recovered_bits)

ber_raw = np.mean(recovered_coded_bits != coded_bits)
ber_ecc = np.mean(recovered_bits != bits)

fig, axes = plt.subplots(1, 3, figsize=(15, 3))
fig.suptitle("Step 9 · Bit Extraction & ECC Decoding", fontsize=12, fontweight="bold")

plot_bits(axes[0], coded_bits,           f"ORIGINAL coded bits  ({len(coded_bits)})")
plot_bits(axes[1], recovered_coded_bits, f"RECOVERED coded bits  (BER={ber_raw*100:.1f}%)")

# Error map
errors = (recovered_coded_bits != coded_bits).astype(float)[:len(coded_bits)]
n = len(errors); cols = min(n, 40); rows = int(np.ceil(n/cols))
grid = np.full(rows*cols, np.nan); grid[:n] = errors; grid = grid.reshape(rows, cols)
axes[2].imshow(grid, cmap="RdYlGn_r", vmin=0, vmax=1, interpolation="nearest", aspect="auto")
axes[2].set_title(f"Error positions (red=flip)  {int(errors.sum())} flips / {n} bits", fontsize=9)
axes[2].axis("off")

plt.tight_layout(); plt.show()
"""),

M("## 14 · ECC Majority Vote Decoding"),

C("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 3))
fig.suptitle("Step 10 · ECC Majority Vote → Final Bits → Text", fontsize=12, fontweight="bold")

plot_bits(axes[0], recovered_bits, f"Final decoded bits  ({len(recovered_bits)})  BER={ber_ecc*100:.1f}%")
plot_bits(axes[1], bits,           "Original bits (reference)")

plt.tight_layout(); plt.show()

print()
print("━"*60)
print(f"  Raw BER (before ECC) : {ber_raw*100:.2f}%")
print(f"  BER after ECC        : {ber_ecc*100:.2f}%")
print(f"  Original message     : {secret_message!r}")
print(f"  Recovered message    : {recovered_message!r}")
match = recovered_message.strip() == secret_message.strip()
print(f"  Exact match          : {'✓ YES' if match else '✗ NO — increase REPS or EMBED_STRENGTH'}")
print("━"*60)
"""),

# ── Section 15: full pipeline summary ─────────────────────────────────────────
M("## 15 · Full Pipeline Summary Dashboard"),

C("""\
fig = plt.figure(figsize=(18, 10))
gs  = gridspec.GridSpec(3, 5, figure=fig, hspace=0.45, wspace=0.35)

def mini_bits(ax, bits, title, max_show=80):
    show = bits[:max_show]
    n = len(show); cols = min(n,40); rows=int(np.ceil(n/cols))
    g = np.full(rows*cols,-1.0); g[:n]=show.astype(float); g=g.reshape(rows,cols)
    ax.imshow(g, cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest", aspect="auto")
    ax.set_title(title, fontsize=7, fontweight="bold"); ax.axis("off")

# Row 0
ax00 = fig.add_subplot(gs[0,0]); mini_bits(ax00, bits, f"1. Secret bits\\n({len(bits)} bits)")
ax01 = fig.add_subplot(gs[0,1]); mini_bits(ax01, coded_bits, f"2. ECC coded bits\\n({len(coded_bits)} bits)")
ax02 = fig.add_subplot(gs[0,2])
ax02.imshow(ch0_arr, cmap="RdBu_r", interpolation="nearest")
ax02.set_title("3. Raw noise\\n(Channel 0)", fontsize=7, fontweight="bold"); ax02.axis("off")
ax03 = fig.add_subplot(gs[0,3])
ax03.imshow(embedded_ch, cmap="RdBu_r", interpolation="nearest")
ax03.set_title("4. Noise + secret\\n(embedded)", fontsize=7, fontweight="bold"); ax03.axis("off")
ax04 = fig.add_subplot(gs[0,4])
ax04.imshow(stego_image)
ax04.set_title("5. Stego image\\n(AI-generated)", fontsize=7, fontweight="bold"); ax04.axis("off")

# Row 1: spectra
ax10 = fig.add_subplot(gs[1,0])
ax10.imshow(Z_mag, cmap="inferno", interpolation="nearest")
ax10.set_title("Spectrum — raw noise", fontsize=7); ax10.axis("off")
ax11 = fig.add_subplot(gs[1,1])
ax11.imshow(Z_emb_mag, cmap="inferno", interpolation="nearest")
ax11.set_title("Spectrum — embedded", fontsize=7); ax11.axis("off")

# Bit pattern in ring
ax12 = fig.add_subplot(gs[1,2])
base2 = np.log1p(np.abs(np.fft.fftshift(Z_emb)))
ax12.imshow(base2, cmap="Greys_r", interpolation="nearest")
ax12.imshow(bit_map, cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest", alpha=0.85)
ax12.set_title("Bits in ring", fontsize=7); ax12.axis("off")

ax13 = fig.add_subplot(gs[1,3])
ax13.imshow(recovered_channel, cmap="RdBu_r", interpolation="nearest")
ax13.set_title("6. Recovered noise\\n(DDIM inversion)", fontsize=7); ax13.axis("off")
ax14 = fig.add_subplot(gs[1,4]); mini_bits(ax14, recovered_coded_bits, f"7. Recovered coded bits\\nBER {ber_raw*100:.1f}%")

# Row 2: summary
ax20 = fig.add_subplot(gs[2,0]); mini_bits(ax20, recovered_bits, f"8. Decoded bits\\nBER {ber_ecc*100:.1f}%")
ax21 = fig.add_subplot(gs[2, 1:3])
ax21.axis("off")
match = recovered_message.strip() == secret_message.strip()
summary = (
    f"ORIGINAL : {secret_message!r}\\n"
    f"RECOVERED: {recovered_message!r}\\n\\n"
    f"Match    : {'✓ YES' if match else '✗ NO'}\\n"
    f"BER raw  : {ber_raw*100:.2f}%\\n"
    f"BER (ECC): {ber_ecc*100:.2f}%\\n"
    f"REPS     : {REPS}     STRENGTH: {EMBED_STRENGTH}"
)
ax21.text(0.05, 0.5, summary, fontsize=10, family="monospace",
          va="center", transform=ax21.transAxes,
          bbox=dict(boxstyle="round", facecolor="#1e3a5f", edgecolor="#4fc3f7", alpha=0.9),
          color="white")
ax21.set_title("Results Summary", fontsize=9, fontweight="bold")

ax22 = fig.add_subplot(gs[2, 3:])
ax22.axis("off")
pipeline = (
    "ENCODING PIPELINE\\n"
    "─────────────────\\n"
    "Text → bits → ECC\\n"
    "  ↓\\n"
    "Gaussian noise (64×64×4)\\n"
    "  ↓  [FFT Channel 0]\\n"
    "Embed bits in ring coefficients\\n"
    "  ↓  [IFFT]\\n"
    "Structured noise\\n"
    "  ↓  [Frozen U-Net 50 steps]\\n"
    "Stego Image (512×512)\\n\\n"
    "DECODING PIPELINE\\n"
    "─────────────────\\n"
    "Stego image\\n"
    "  ↓  [Frozen VAE encoder]\\n"
    "Latent z₀\\n"
    "  ↓  [DDIM inversion 50 steps]\\n"
    "Recovered noise\\n"
    "  ↓  [FFT → ring → signs]\\n"
    "Coded bits → ECC → text"
)
ax22.text(0.05, 0.95, pipeline, fontsize=8, family="monospace",
          va="top", transform=ax22.transAxes,
          bbox=dict(boxstyle="round", facecolor="#1a3a1a", edgecolor="#4caf50", alpha=0.9),
          color="#e8f5e9")

fig.suptitle("Complete Generative Steganography Pipeline — Visual Summary",
             fontsize=13, fontweight="bold", y=1.01)
plt.savefig("steganography_summary.png", dpi=150, bbox_inches="tight")
plt.show()
print("✓ Summary saved to steganography_summary.png")
"""),
]

# ─────────────────────────────────────────────────────────────────────────────
# NOTEBOOK 2 — VS Code Python script: Decoder
# ─────────────────────────────────────────────────────────────────────────────

decoder_script = '''\
"""
decode_secret.py  — VS Code / terminal decoder for Generative Steganography
=============================================================================
Run:  python decode_secret.py

Requires in the same folder:
  • stego_image.png          (output of the embedding notebook)
  • coeff_order.npy
  • coded_bits_original.npy  (optional — used only for accuracy metrics)
  • bits_original.npy        (optional — used only for accuracy metrics)

The script visualises every step of the decoding pipeline and prints the
recovered secret message.
"""

import torch, numpy as np, matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler
from PIL import Image
import warnings, os, sys
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# 0. Configuration — must match what was used during embedding
# ──────────────────────────────────────────────────────────────────────────────
MODEL_ID       = "runwayml/stable-diffusion-v1-5"
PROMPT         = "a scenic mountain landscape at sunset, highly detailed, oil painting"
NUM_STEPS      = 50
GUIDANCE_SCALE = 1.0
REPS           = 5
RING_R_LOW     = 6
RING_R_HIGH    = 28
EMBED_CHANNEL  = 0
SEED_KEY       = 1234
STEGO_PATH     = "stego_image.png"
COEFF_PATH     = "coeff_order.npy"
ORIG_CODED_PATH= "coded_bits_original.npy"   # optional
ORIG_BITS_PATH = "bits_original.npy"          # optional

# ──────────────────────────────────────────────────────────────────────────────
# 1. Helper functions
# ──────────────────────────────────────────────────────────────────────────────
def bits_to_text(bits):
    n = (len(bits) // 8) * 8
    b = np.packbits(bits[:n].astype(np.uint8))
    return b.tobytes().decode("utf-8", errors="replace")

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
        if (my, mx) in seen:    continue
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
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.axis("off")
    if highlight_reps and highlight_reps > 1:
        for col in range(0, min(cols, n), highlight_reps):
            ax.axvline(col - 0.5, color="#4fc3f7", linewidth=0.8, alpha=0.7)

# ──────────────────────────────────────────────────────────────────────────────
# 2. Load required files
# ──────────────────────────────────────────────────────────────────────────────
print("="*65)
print("  GENERATIVE STEGANOGRAPHY — SECRET DECODER")
print("="*65)

if not os.path.exists(STEGO_PATH):
    sys.exit(f"ERROR: {STEGO_PATH!r} not found. Run the embedding notebook first.")
if not os.path.exists(COEFF_PATH):
    sys.exit(f"ERROR: {COEFF_PATH!r} not found.")

stego_image  = Image.open(STEGO_PATH)
coeff_order  = np.load(COEFF_PATH)
n_coded_bits = len(coeff_order)
n_orig_bits  = (n_coded_bits // REPS)

# Optional ground-truth for metrics
have_gt = os.path.exists(ORIG_CODED_PATH) and os.path.exists(ORIG_BITS_PATH)
if have_gt:
    orig_coded = np.load(ORIG_CODED_PATH)
    orig_bits  = np.load(ORIG_BITS_PATH)
    print(f"Ground-truth loaded   ✓  ({n_orig_bits} original bits)")
else:
    print("Ground-truth not found — accuracy metrics will be skipped")

print(f"Stego image loaded    ✓  {stego_image.size}")
print(f"Coefficient order     ✓  {len(coeff_order)} positions")

# ──────────────────────────────────────────────────────────────────────────────
# 3. STEP VIS: show stego image
# ──────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(stego_image)
ax.set_title("DECODER · Step 1\\nStego image received — secret is hidden inside",
             fontsize=11, fontweight="bold")
ax.axis("off")
plt.tight_layout(); plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# 4. Load model
# ──────────────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if device == "cuda" else torch.float32
print(f"\\nDevice : {device} | dtype : {dtype}")
print("Loading frozen Stable Diffusion model …")

pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype, safety_checker=None)
pipe = pipe.to(device)
inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
for m in [pipe.unet, pipe.vae, pipe.text_encoder]:
    m.requires_grad_(False); m.eval()

LATENT_CHANNELS = pipe.unet.config.in_channels
LATENT_SIZE     = pipe.unet.config.sample_size
VAE_SCALE       = pipe.vae.config.scaling_factor
print(f"Model loaded  ✓  Latent: ({LATENT_CHANNELS},{LATENT_SIZE},{LATENT_SIZE})")

# Ring coordinates
ring_coords = get_ring_coords(LATENT_SIZE, LATENT_SIZE, RING_R_LOW, RING_R_HIGH)

# ──────────────────────────────────────────────────────────────────────────────
# 5. STEP VIS: VAE encode stego → z₀
# ──────────────────────────────────────────────────────────────────────────────
print("\\nStep 2: VAE encoding stego image → latent z₀ …")
with torch.no_grad():
    img_t = pipe.image_processor.preprocess(stego_image).to(device=device, dtype=dtype)
    z0_hat = pipe.vae.encode(img_t).latent_dist.mode() * VAE_SCALE

fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
fig.suptitle("DECODER · Step 2 · VAE Encoded Latent z₀ (all 4 channels)",
             fontsize=11, fontweight="bold")
for ch in range(4):
    data = z0_hat[0, ch].float().cpu().numpy()
    im = axes[ch].imshow(data, cmap="RdBu_r", interpolation="nearest")
    axes[ch].set_title(f"Channel {ch}  σ={data.std():.2f}", fontsize=9)
    axes[ch].axis("off")
    plt.colorbar(im, ax=axes[ch], shrink=0.85)
plt.tight_layout(); plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# 6. DDIM inversion with snapshots
# ──────────────────────────────────────────────────────────────────────────────
print("Step 3: DDIM inversion (tracing image → noise) …")
with torch.no_grad():
    prompt_embeds, _ = pipe.encode_prompt(
        PROMPT, device, num_images_per_prompt=1,
        do_classifier_free_guidance=False, negative_prompt=None,
    )
    inverse_scheduler.set_timesteps(NUM_STEPS, device=device)
    inv_latents = z0_hat
    snapshots   = {}
    snap_steps  = {0, NUM_STEPS//4, NUM_STEPS//2, 3*NUM_STEPS//4, NUM_STEPS-1}

    for step_i, t in enumerate(inverse_scheduler.timesteps):
        noise_pred  = pipe.unet(inv_latents, t, encoder_hidden_states=prompt_embeds).sample
        inv_latents = inverse_scheduler.step(noise_pred, t, inv_latents).prev_sample
        if step_i in snap_steps:
            snapshots[step_i] = inv_latents[0, EMBED_CHANNEL].float().cpu().numpy().copy()

print("Inversion complete ✓")

# ──────────────────────────────────────────────────────────────────────────────
# 7. STEP VIS: inversion progress
# ──────────────────────────────────────────────────────────────────────────────
snaps = sorted(snapshots.keys())
fig, axes = plt.subplots(2, len(snaps), figsize=(15, 6))
fig.suptitle("DECODER · Step 3 · DDIM Inversion Progress — Recovering Noise from Image",
             fontsize=11, fontweight="bold")
for col, si in enumerate(snaps):
    data = snapshots[si]
    axes[0, col].imshow(data, cmap="RdBu_r", interpolation="nearest")
    axes[0, col].set_title(f"Step {si+1}/{NUM_STEPS}\\nσ={data.std():.2f}", fontsize=9)
    axes[0, col].axis("off")
    Z = np.fft.fftshift(np.log1p(np.abs(np.fft.fft2(data))))
    axes[1, col].imshow(Z, cmap="inferno", interpolation="nearest")
    axes[1, col].set_title("Spectrum", fontsize=8); axes[1, col].axis("off")
plt.tight_layout(); plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# 8. Extract bits from recovered noise
# ──────────────────────────────────────────────────────────────────────────────
recovered_channel    = inv_latents[0, EMBED_CHANNEL].float().cpu().numpy()
recovered_coded_bits = extract_bits_from_channel(
    recovered_channel, ring_coords, coeff_order, n_coded_bits
)
recovered_bits    = ecc_decode(recovered_coded_bits, n_orig_bits, REPS)
recovered_message = bits_to_text(recovered_bits)

# ──────────────────────────────────────────────────────────────────────────────
# 9. STEP VIS: FFT of recovered noise + ring read-out
# ──────────────────────────────────────────────────────────────────────────────
Z_rec = np.fft.fft2(recovered_channel)

# Build bit-position overlay
bit_map_rec = np.full((LATENT_SIZE, LATENT_SIZE), np.nan)
for i, idx in enumerate(coeff_order[:n_coded_bits]):
    y, x  = ring_coords[idx]
    sy    = (y + LATENT_SIZE // 2) % LATENT_SIZE
    sx    = (x + LATENT_SIZE // 2) % LATENT_SIZE
    bit_map_rec[sy, sx] = recovered_coded_bits[i]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("DECODER · Step 4 · FFT of Recovered Noise — Reading Bits from Ring",
             fontsize=11, fontweight="bold")

base = np.log1p(np.abs(np.fft.fftshift(Z_rec)))
axes[0].imshow(base, cmap="Greys_r", interpolation="nearest")
sc = axes[0].imshow(bit_map_rec, cmap="RdYlGn", vmin=0, vmax=1,
                    interpolation="nearest", alpha=0.9)
axes[0].set_title("Recovered bits overlaid on spectrum\\n(green=1, red=0)", fontsize=9)
plt.colorbar(sc, ax=axes[0], shrink=0.8, label="bit value")
from matplotlib.patches import Circle
cx = cy = LATENT_SIZE // 2
for r, ls in [(RING_R_LOW,"--"),(RING_R_HIGH,"-")]:
    axes[0].add_patch(Circle((cx,cy), r, fill=False, edgecolor="#4fc3f7", lw=1.2, ls=ls))

real_parts = [Z_rec[ring_coords[idx][0], ring_coords[idx][1]].real for idx in coeff_order[:n_coded_bits]]
axes[1].bar(range(len(real_parts)), real_parts,
            color=["#4caf50" if v>0 else "#f44336" for v in real_parts],
            alpha=0.8, width=1.0)
axes[1].axhline(0, color="white", lw=0.8)
axes[1].set_xlabel("Bit position", fontsize=9)
axes[1].set_ylabel("Re(Z[y,x])", fontsize=9)
axes[1].set_title("Sign → bit:  + → 1,  − → 0", fontsize=9, fontweight="bold")
axes[1].grid(alpha=0.2)

plot_bits(axes[2], recovered_coded_bits, f"Extracted coded bits ({n_coded_bits})")
plt.tight_layout(); plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# 10. STEP VIS: ECC decode
# ──────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 3))
fig.suptitle("DECODER · Step 5 · ECC Majority-Vote Decoding → Final Bits",
             fontsize=11, fontweight="bold")
plot_bits(axes[0], recovered_coded_bits,
          f"Coded bits (x{REPS} redundancy, {n_coded_bits} total)", highlight_reps=REPS)
plot_bits(axes[1], recovered_bits, f"Decoded bits ({n_orig_bits}) after majority vote")
plt.tight_layout(); plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# 11. STEP VIS: accuracy metrics (if ground truth available)
# ──────────────────────────────────────────────────────────────────────────────
if have_gt:
    ber_raw = np.mean(recovered_coded_bits != orig_coded)
    ber_ecc = np.mean(recovered_bits != orig_bits)

    fig, axes = plt.subplots(1, 3, figsize=(15, 3))
    fig.suptitle("DECODER · Step 6 · Bit Error Analysis", fontsize=11, fontweight="bold")

    plot_bits(axes[0], orig_coded, f"ORIGINAL coded bits ({len(orig_coded)})")
    plot_bits(axes[1], recovered_coded_bits, f"RECOVERED coded bits  BER={ber_raw*100:.1f}%")
    errors = (recovered_coded_bits[:len(orig_coded)] != orig_coded).astype(float)
    n = len(errors); cols = min(n,40); rows = int(np.ceil(n/cols))
    g = np.full(rows*cols, np.nan); g[:n] = errors; g = g.reshape(rows,cols)
    axes[2].imshow(g, cmap="RdYlGn_r", vmin=0, vmax=1, interpolation="nearest", aspect="auto")
    axes[2].set_title(f"Flip positions (red=error)  {int(errors.sum())} / {n}", fontsize=9)
    axes[2].axis("off")
    plt.tight_layout(); plt.show()
else:
    ber_raw = ber_ecc = None

# ──────────────────────────────────────────────────────────────────────────────
# 12. Final summary dashboard
# ──────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 9))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)

ax0 = fig.add_subplot(gs[0, 0])
ax0.imshow(stego_image); ax0.axis("off")
ax0.set_title("Stego image (input)", fontsize=9, fontweight="bold")

ax1 = fig.add_subplot(gs[0, 1])
ax1.imshow(recovered_channel, cmap="RdBu_r", interpolation="nearest"); ax1.axis("off")
ax1.set_title("Recovered noise (Ch 0)\\nvia DDIM inversion", fontsize=9, fontweight="bold")

ax2 = fig.add_subplot(gs[0, 2])
base2 = np.log1p(np.abs(np.fft.fftshift(Z_rec)))
ax2.imshow(base2, cmap="Greys_r", interpolation="nearest")
ax2.imshow(bit_map_rec, cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest", alpha=0.85)
ax2.set_title("Bits in frequency ring", fontsize=9, fontweight="bold"); ax2.axis("off")

def mini(ax, bits, title):
    n=len(bits[:80]); cols=min(n,40); rows=int(np.ceil(n/cols))
    g=np.full(rows*cols,-1.0); g[:n]=bits[:n].astype(float); g=g.reshape(rows,cols)
    ax.imshow(g, cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest", aspect="auto")
    ax.set_title(title, fontsize=8, fontweight="bold"); ax.axis("off")

ax3 = fig.add_subplot(gs[1, 0]); mini(ax3, recovered_coded_bits, "Extracted coded bits")
ax4 = fig.add_subplot(gs[1, 1]); mini(ax4, recovered_bits, "Decoded final bits")

ax5 = fig.add_subplot(gs[1, 2]); ax5.axis("off")
lines = [
    "DECODING RESULT",
    "─"*32,
    f"Recovered : {recovered_message!r}",
]
if have_gt:
    match = recovered_message.strip() == bits_to_text(orig_bits).strip()
    lines += [
        f"BER raw   : {ber_raw*100:.2f}%",
        f"BER (ECC) : {ber_ecc*100:.2f}%",
        f"Match     : {'✓ YES' if match else '✗ NO'}",
    ]
else:
    lines.append("(ground-truth not loaded)")
ax5.text(0.05, 0.5, "\\n".join(lines), fontsize=11, family="monospace",
         va="center", transform=ax5.transAxes,
         bbox=dict(boxstyle="round", facecolor="#1e3a5f", edgecolor="#4fc3f7", alpha=0.9),
         color="white")

fig.suptitle("Generative Steganography — Decoder Summary Dashboard",
             fontsize=13, fontweight="bold")
plt.savefig("decoder_summary.png", dpi=150, bbox_inches="tight")
plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# 13. Print final answer
# ──────────────────────────────────────────────────────────────────────────────
print()
print("="*65)
print(f"  RECOVERED SECRET MESSAGE: {recovered_message!r}")
if have_gt:
    print(f"  Raw BER  : {ber_raw*100:.2f}%   |   BER after ECC: {ber_ecc*100:.2f}%")
print("  Decoder summary saved → decoder_summary.png")
print("="*65)
'''

# ─────────────────────────────────────────────────────────────────────────────
# Write notebook 1
# ─────────────────────────────────────────────────────────────────────────────
nb = {
    "cells": embed_cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"}
    },
    "nbformat": 4, "nbformat_minor": 5
}
with open("pretrained_generative_steganography.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("✓ Notebook 1 written")

# Write script 2
with open("decode_secret.py", "w") as f:
    f.write(decoder_script)
print("✓ Script 2  written")