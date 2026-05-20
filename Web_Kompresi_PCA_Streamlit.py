import streamlit as st
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from scipy.ndimage import convolve
from skimage.metrics import structural_similarity

# ════════════════════════════════════════════════════════════════
# KONFIGURASI HALAMAN
# ════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="PCA Image Compressor",
    page_icon="🏰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .section-header {
        background: #1F3864;
        color: #FFD700;
        padding: 10px 18px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 16px;
        margin: 20px 0 10px 0;
    }
    .metrik-box {
        background: #f0f4ff;
        border-left: 4px solid #2E75B6;
        padding: 8px 14px;
        border-radius: 4px;
        margin: 4px 0;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# FUNGSI ANALISIS GAMBAR
# ════════════════════════════════════════════════════════════════

def hitung_sharpness(arr):
    """Ketajaman via variansi Laplacian dari kanal luminansi."""
    gray = (0.2126 * arr[:,:,0] + 0.7152 * arr[:,:,1]
            + 0.0722 * arr[:,:,2]).astype(np.float32)
    kernel = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=np.float32)
    return float(np.var(convolve(gray, kernel)))

def hitung_entropy(arr):
    """Entropi Shannon dari kanal luminansi."""
    gray = (0.2126 * arr[:,:,0] + 0.7152 * arr[:,:,1]
            + 0.0722 * arr[:,:,2]).astype(np.uint8)
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0,255))
    hist = hist.astype(float)
    hist /= hist.sum()
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))

def hitung_colorfulness(arr):
    """Tingkat keberagaman warna — metrik Hasler & Süsstrunk."""
    r, g, b = arr[:,:,0].astype(float), arr[:,:,1].astype(float), arr[:,:,2].astype(float)
    rg = r - g
    yb = 0.5 * (r + g) - b
    return math.sqrt(np.std(rg)**2 + np.std(yb)**2) + 0.3 * math.sqrt(np.mean(rg)**2 + np.mean(yb)**2)


# ════════════════════════════════════════════════════════════════
# IMPLEMENTASI PCA MANUAL — 6 LANGKAH MATEMATIS
# ════════════════════════════════════════════════════════════════

def pca_manual(matriks, k):
    """
    PCA 6-langkah untuk satu matriks 2D (satu kanal gambar).
    Return: (Xhat, total_var, eigenvalues_sorted, k_aktual)
    """
    # Langkah 1 — Centering: Xc = X − X̄
    mean_kolom = np.mean(matriks, axis=0)
    Xc = matriks - mean_kolom

    # Langkah 2 — Matriks Kovarians: C = XcᵀXc / (n−1)
    n = Xc.shape[0]
    C = (Xc.T @ Xc) / (n - 1)

    # Langkah 3 — Dekomposisi Eigen: C·v = λ·v (eigh untuk matriks simetris)
    eigenvalues, eigenvectors = np.linalg.eigh(C)

    # Langkah 4 — Urut descending: λ₁ > λ₂ > ... > λ_d
    urutan = np.argsort(eigenvalues)[::-1]
    eigenvalues  = eigenvalues[urutan]
    eigenvectors = eigenvectors[:, urutan]

    # Langkah 5 — Pilih K komponen: W = V[:, :k]
    k_aman = min(k, C.shape[0])
    W = eigenvectors[:, :k_aman]
    total_var = float(np.sum(eigenvalues[:k_aman]) / np.sum(np.abs(eigenvalues)))

    # Langkah 6 — Proyeksi & Rekonstruksi: Z = Xc·W, X̂ = Z·Wᵀ + X̄
    Z    = Xc @ W
    Xhat = (Z @ W.T) + mean_kolom

    return Xhat, total_var, eigenvalues, k_aman


# ════════════════════════════════════════════════════════════════
# FUNGSI KOMPRESI
# ════════════════════════════════════════════════════════════════

def kompresi_warna(arr, k):
    """PCA dijalankan 3× secara independen: kanal R, G, B."""
    hasil_kanal, eigenvalues_list, var_list = [], [], []
    for i in range(3):
        rekon, var, ev, k_a = pca_manual(arr[:,:,i].astype(float), k)
        hasil_kanal.append(rekon)
        eigenvalues_list.append(ev)
        var_list.append(var)

    arr_hasil = np.clip(np.dstack(hasil_kanal), 0, 255).astype(np.uint8)
    min_len   = min(len(e) for e in eigenvalues_list)
    ev_avg    = np.mean([e[:min_len] for e in eigenvalues_list], axis=0)
    return arr_hasil, float(np.mean(var_list)), ev_avg, k_a

def kompresi_grayscale(arr, k):
    """
    Konversi ke luminansi (ITU-R BT.601) → PCA dijalankan 1×.
    Bobot: 0.2126R + 0.7152G + 0.0722B (sensitivitas mata manusia).
    """
    lum = (0.2126 * arr[:,:,0] + 0.7152 * arr[:,:,1]
           + 0.0722 * arr[:,:,2]).astype(float)
    rekon, var, ev, k_a = pca_manual(lum, k)
    lum_u8  = np.clip(rekon, 0, 255).astype(np.uint8)
    arr_out = np.stack([lum_u8] * 3, axis=-1)
    return arr_out, var, ev, k_a

def arr_ke_grayscale(arr):
    """Konversi RGB → grayscale 3-kanal (untuk perbandingan visual)."""
    lum = (0.2126*arr[:,:,0] + 0.7152*arr[:,:,1]
           + 0.0722*arr[:,:,2]).astype(np.uint8)
    return np.stack([lum]*3, axis=-1)


# ════════════════════════════════════════════════════════════════
# METRIK EVALUASI
# ════════════════════════════════════════════════════════════════

def hitung_metrik(arr_asli, arr_komp):
    """Hitung MSE, PSNR, dan SSIM."""
    a = arr_asli.astype(float)
    b = arr_komp.astype(float)
    mse  = float(np.mean((a - b) ** 2))
    psnr = 20 * math.log10(255.0 / math.sqrt(mse)) if mse > 0 else float('inf')
    ssim_val = structural_similarity(
        arr_asli, arr_komp, channel_axis=-1, data_range=255
    )
    return mse, psnr, float(ssim_val)

def hitung_rasio(tinggi, lebar, k):
    """Estimasi rasio memori: (H·k + k·W)×3 / (H·W×3) × 100%"""
    asli   = tinggi * lebar * 3
    simpan = ((tinggi * k) + (k * lebar)) * 3
    return (simpan / asli) * 100


# ════════════════════════════════════════════════════════════════
# FUNGSI PLOTTING
# ════════════════════════════════════════════════════════════════

def plot_histogram_rgb(arr, judul, figsize=(5.5, 3)):
    fig, ax = plt.subplots(figsize=figsize)
    warna = ['#e74c3c', '#27ae60', '#2980b9']
    label = ['R', 'G', 'B']
    for i in range(3):
        ax.hist(arr[:,:,i].flatten(), bins=128, range=(0,255),
                color=warna[i], alpha=0.55, label=label[i])
    ax.set_title(judul, fontsize=10, fontweight='bold')
    ax.set_xlabel('Intensitas Piksel (0–255)')
    ax.set_ylabel('Frekuensi')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_histogram_gray(arr, judul, figsize=(5.5, 3)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(arr[:,:,0].flatten(), bins=128, range=(0,255), color='#555', alpha=0.8)
    ax.set_title(judul, fontsize=10, fontweight='bold')
    ax.set_xlabel('Intensitas Piksel (0–255)')
    ax.set_ylabel('Frekuensi')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_scree(eigenvalues, k_dipilih, judul, figsize=(5.5, 3.5)):
    tampil = min(60, len(eigenvalues))
    ev = eigenvalues[:tampil]
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(range(1, tampil+1), ev, 'o-', color='#2980b9', markersize=3.5, linewidth=1.5)
    if k_dipilih <= tampil:
        ax.axvline(x=k_dipilih, color='#e74c3c', linestyle='--', linewidth=1.5,
                   label=f'K = {k_dipilih}')
        ax.legend(fontsize=9)
    ax.set_title(judul, fontsize=10, fontweight='bold')
    ax.set_xlabel('Nomor Komponen (PC)')
    ax.set_ylabel('Eigenvalue (λ)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_cumvar(eigenvalues, k_dipilih, judul, figsize=(5.5, 3.5)):
    cum_var  = np.cumsum(eigenvalues) / np.sum(np.abs(eigenvalues)) * 100
    tampil   = min(len(eigenvalues), max(k_dipilih + 30, 60))
    fig, ax  = plt.subplots(figsize=figsize)
    ax.plot(range(1, tampil+1), cum_var[:tampil], '-', color='#8e44ad', linewidth=2)
    for pct, clr in zip([90, 95, 99], ['#f39c12', '#27ae60', '#e74c3c']):
        ax.axhline(y=pct, color=clr, linestyle=':', linewidth=1.2, label=f'{pct}%')
    if k_dipilih <= tampil:
        ax.axvline(x=k_dipilih, color='#2980b9', linestyle='--', linewidth=1.5,
                   label=f'K={k_dipilih} → {cum_var[k_dipilih-1]:.1f}%')
    ax.set_title(judul, fontsize=10, fontweight='bold')
    ax.set_xlabel('Jumlah Komponen K')
    ax.set_ylabel('Variansi Kumulatif (%)')
    ax.set_ylim([0, 105])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_error_image(arr_asli, arr_komp, judul, figsize=(4.5, 3.5)):
    error     = np.abs(arr_asli.astype(float) - arr_komp.astype(float))
    error_lum = np.mean(error, axis=2)
    fig, ax   = plt.subplots(figsize=figsize)
    im = ax.imshow(error_lum, cmap='hot', vmin=0, vmax=error_lum.max() or 1)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(judul, fontsize=10, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════
# TABEL BENCHMARK MULTI-K
# ════════════════════════════════════════════════════════════════

def nilai_k_patokan(k_maks, k_user):
    """
    Buat ~6 nilai K yang tersebar proporsional di antara 1 dan k_maks,
    ditambah K yang sedang dipilih pengguna.
    """
    patokan = sorted(set([
        max(1, k_maks //  20),
        max(1, k_maks //  10),
        max(1, k_maks //   5),
        max(1, k_maks //   3),
        max(1, k_maks //   2),
        max(1, k_maks *  3 // 4),
        k_maks,
        k_user,
    ]))
    return patokan

@st.cache_data(show_spinner=False)
def benchmark_warna(arr_bytes, shape, k_user):
    arr    = np.frombuffer(arr_bytes, dtype=np.uint8).reshape(shape)
    h, w, _ = arr.shape
    k_maks  = min(h, w)
    rows = []
    for k in nilai_k_patokan(k_maks, k_user):
        arr_k, var, _, _ = kompresi_warna(arr, k)
        mse, psnr, ssim_v = hitung_metrik(arr, arr_k)
        rows.append({
            'K': k,
            'EVR (%)': round(var * 100, 1),
            'MSE': round(mse, 2),
            'PSNR (dB)': round(psnr, 2),
            'SSIM': round(ssim_v, 4),
            'Rasio Memori (%)': round(hitung_rasio(h, w, k), 1),
            'Ket.': '⬅ dipilih' if k == k_user else ''
        })
    return rows

@st.cache_data(show_spinner=False)
def benchmark_grayscale(arr_bytes, shape, k_user):
    arr     = np.frombuffer(arr_bytes, dtype=np.uint8).reshape(shape)
    arr_ref = arr_ke_grayscale(arr)
    h, w, _ = arr.shape
    k_maks  = min(h, w)
    rows = []
    for k in nilai_k_patokan(k_maks, k_user):
        arr_k, var, _, _ = kompresi_grayscale(arr, k)
        mse, psnr, ssim_v = hitung_metrik(arr_ref, arr_k)
        rows.append({
            'K': k,
            'EVR (%)': round(var * 100, 1),
            'MSE': round(mse, 2),
            'PSNR (dB)': round(psnr, 2),
            'SSIM': round(ssim_v, 4),
            'Rasio Memori (%)': round(hitung_rasio(h, w, k), 1),
            'Ket.': '⬅ dipilih' if k == k_user else ''
        })
    return rows


# ════════════════════════════════════════════════════════════════
# HELPER — tampilkan satu section hasil (warna / grayscale)
# ════════════════════════════════════════════════════════════════

def tampilkan_section_hasil(
    arr_asli_visual,   # gambar "asli" untuk ditampilkan (RGB atau gray-3ch)
    arr_kompresi,      # gambar hasil kompresi
    eigenvalues,       # untuk scree & cumvar
    k, mse, psnr, ssim_v, var, rasio,
    label_mode,        # "Warna" atau "Grayscale"
    fn_histogram,      # plot_histogram_rgb / plot_histogram_gray
):
    # ── Tiga gambar berdampingan ──────────────────────────────
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.image(arr_asli_visual, caption="Gambar Asli", width="stretch")
    with c2:
        st.image(arr_kompresi, caption=f"Hasil Kompresi (K={k})", width="stretch")
    with c3:
        fig_err = plot_error_image(arr_asli_visual, arr_kompresi, "|X − X̂| Error Image")
        st.pyplot(fig_err); plt.close(fig_err)

    # ── Metrik utama ──────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("📊 EVR",         f"{var*100:.2f}%")
    m2.metric("📉 MSE",         f"{mse:.3f}")
    m3.metric("📶 PSNR",        f"{psnr:.2f} dB")
    m4.metric("🔍 SSIM",        f"{ssim_v:.4f}")
    m5.metric("💾 Rasio Memori", f"{rasio:.1f}%")

    # ── Scree + Cumulative Variance ───────────────────────────
    g1, g2 = st.columns(2)
    with g1:
        fig_s = plot_scree(eigenvalues, k, f"Scree Plot — {label_mode}")
        st.pyplot(fig_s); plt.close(fig_s)
    with g2:
        fig_c = plot_cumvar(eigenvalues, k, f"Cumulative Explained Variance — {label_mode}")
        st.pyplot(fig_c); plt.close(fig_c)

    # ── Histogram sebelum vs sesudah ──────────────────────────
    h1, h2 = st.columns(2)
    with h1:
        fig_ha = fn_histogram(arr_asli_visual, "Histogram — Sebelum Kompresi")
        st.pyplot(fig_ha); plt.close(fig_ha)
    with h2:
        fig_hb = fn_histogram(arr_kompresi, f"Histogram — Sesudah Kompresi (K={k})")
        st.pyplot(fig_hb); plt.close(fig_hb)


# ════════════════════════════════════════════════════════════════
# ANTARMUKA UTAMA STREAMLIT
# ════════════════════════════════════════════════════════════════

st.title("🏰 PCA Image Compressor")
st.caption("Kompresi gambar menggunakan Principal Component Analysis — implementasi manual NumPy | 6 Langkah Matematis")
st.markdown("---")

# ── Upload gambar ─────────────────────────────────────────────
uploaded = st.file_uploader(
    "📁 Upload Gambar (JPG / PNG / BMP / WEBP)",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded is None:
    st.info("👆 Upload gambar untuk memulai.")
    st.stop()

# ── Baca gambar ───────────────────────────────────────────────
MAX_DIM = 900   # batas maksimum dimensi terpanjang (px) agar tidak OOM

img = Image.open(uploaded).convert("RGB")
w_asli, h_asli = img.size

# Auto-resize jika gambar terlalu besar
if max(w_asli, h_asli) > MAX_DIM:
    ratio   = MAX_DIM / max(w_asli, h_asli)
    w_baru  = int(w_asli * ratio)
    h_baru  = int(h_asli * ratio)
    img     = img.resize((w_baru, h_baru), Image.LANCZOS)
    st.warning(
        f"⚠️ Gambar asli ({w_asli}×{h_asli} px) terlalu besar untuk diproses di server cloud. "
        f"Gambar otomatis di-resize menjadi **{w_baru}×{h_baru} px** agar tidak kehabisan RAM. "
        f"Hasil kompresi tetap representatif."
    )

arr    = np.array(img, dtype=np.uint8)
h, w, _ = arr.shape
k_maks   = min(h, w)

# ── Slider K ──────────────────────────────────────────────────
st.markdown(f"**⚡ Nilai K (Principal Components)** — Maks untuk gambar ini: `{k_maks}`")
k_val = st.slider(
    label="Pilih K",
    min_value=1,
    max_value=min(k_maks, 500),
    value=min(50, k_maks),
    label_visibility="collapsed"
)
st.caption(
    f"K = {k_val} | "
    f"Patokan otomatis: {[v for v in nilai_k_patokan(k_maks, k_val) if v != k_val]} + **{k_val} (dipilih)**"
)

# ════════════════════════════════════════════════════════════════
# BAGIAN 1 — EDA AWAL (langsung tampil setelah upload)
# ════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">📊 Tahap 1 — EDA Awal: Sebelum Kompresi</div>', unsafe_allow_html=True)

col_img, col_stat, col_hist = st.columns([1, 1, 1.5])

with col_img:
    st.image(arr, caption=f"Gambar Asli — {w}×{h} px", width="stretch")
    st.metric("Total Piksel", f"{h*w:,}")
    st.metric("K Maksimum", k_maks)
    lum_global = 0.2126*arr[:,:,0] + 0.7152*arr[:,:,1] + 0.0722*arr[:,:,2]
    st.metric("Kecerahan (Lum.)", f"{np.mean(lum_global):.1f} / 255")
    st.metric("Kontras (Std Lum.)", f"{np.std(lum_global):.1f}")
    st.metric("Entropi Shannon", f"{hitung_entropy(arr):.3f} bit")
    st.metric("Colorfulness", f"{hitung_colorfulness(arr):.2f}")
    try:
        st.metric("Ketajaman (Lap. Var.)", f"{hitung_sharpness(arr):.1f}")
    except Exception:
        pass

with col_stat:
    st.markdown("**Statistik Per-Kanal**")
    nama_kanal = ['🔴 Merah (R)', '🟢 Hijau (G)', '🔵 Biru (B)']
    stat_rows  = []
    for i, nm in enumerate(nama_kanal):
        ch = arr[:,:,i].astype(float)
        stat_rows.append({
            'Kanal':  nm,
            'Mean':   round(float(np.mean(ch)), 2),
            'Std':    round(float(np.std(ch)),  2),
            'Min':    int(np.min(ch)),
            'Max':    int(np.max(ch)),
            'Median': round(float(np.median(ch)), 2),
        })
    st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)

    # Interpretasi otomatis
    mean_lum = float(np.mean(lum_global))
    std_lum  = float(np.std(lum_global))
    st.markdown("**Interpretasi Otomatis**")
    kecerahan_label = "Gelap" if mean_lum < 85 else ("Terang" if mean_lum > 170 else "Normal")
    kontras_label   = "Rendah" if std_lum < 40 else ("Tinggi" if std_lum > 80 else "Sedang")
    st.info(
        f"Kecerahan: **{kecerahan_label}** (mean lum = {mean_lum:.1f})  \n"
        f"Kontras: **{kontras_label}** (std lum = {std_lum:.1f})  \n"
        f"💡 Variansi tinggi → PCA bisa menangkap lebih banyak informasi per-komponen."
    )

with col_hist:
    fig_eda_hist = plot_histogram_rgb(arr, "Histogram Piksel — Gambar Asli (R/G/B)")
    st.pyplot(fig_eda_hist); plt.close(fig_eda_hist)

    arr_gray_asli = arr_ke_grayscale(arr)
    fig_eda_gray  = plot_histogram_gray(arr_gray_asli, "Histogram Luminansi — Representasi Grayscale")
    st.pyplot(fig_eda_gray); plt.close(fig_eda_gray)

# ════════════════════════════════════════════════════════════════
# TOMBOL KOMPRESI
# ════════════════════════════════════════════════════════════════

st.markdown("---")
if st.button("▶  Jalankan Kompresi PCA", type="primary", use_container_width=True):
    with st.spinner("Menjalankan 6 langkah PCA..."):
        arr_c, var_c, ev_c, _ = kompresi_warna(arr, k_val)
        arr_g, var_g, ev_g, _ = kompresi_grayscale(arr, k_val)
        st.session_state['hasil'] = {
            'arr_c': arr_c, 'arr_g': arr_g,
            'var_c': var_c, 'var_g': var_g,
            'ev_c': ev_c,   'ev_g': ev_g,
            'k': k_val,
        }

# ── Tampilkan hasil jika sudah ada di session_state ──────────
if 'hasil' not in st.session_state:
    st.stop()

h_state  = st.session_state['hasil']
arr_c    = h_state['arr_c']
arr_g    = h_state['arr_g']
var_c    = h_state['var_c']
var_g    = h_state['var_g']
ev_c     = h_state['ev_c']
ev_g     = h_state['ev_g']
k        = h_state['k']

arr_gray_asli = arr_ke_grayscale(arr)
mse_c, psnr_c, ssim_c = hitung_metrik(arr,            arr_c)
mse_g, psnr_g, ssim_g = hitung_metrik(arr_gray_asli,  arr_g)
rasio = hitung_rasio(h, w, k)

# ════════════════════════════════════════════════════════════════
# BAGIAN 2 — HASIL KOMPRESI: MODE WARNA
# ════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">🎨 Tahap 2 — Hasil Kompresi: Mode Warna (RGB)</div>', unsafe_allow_html=True)
st.caption("PCA dijalankan **3× secara independen** untuk kanal R, G, dan B. Eigenvalue dirata-rata untuk keperluan plotting.")

tampilkan_section_hasil(
    arr_asli_visual=arr,
    arr_kompresi=arr_c,
    eigenvalues=ev_c,
    k=k, mse=mse_c, psnr=psnr_c, ssim_v=ssim_c,
    var=var_c, rasio=rasio,
    label_mode="Warna (Rata-rata R/G/B)",
    fn_histogram=plot_histogram_rgb,
)

# Tabel benchmark warna
with st.expander("📋 Tabel Benchmark Multi-K — Mode Warna", expanded=True):
    arr_bytes = arr.tobytes()
    with st.spinner("Menghitung benchmark warna..."):
        rows_c = benchmark_warna(arr_bytes, arr.shape, k)
    df_c = pd.DataFrame(rows_c)
    st.dataframe(
        df_c.style.apply(
            lambda col: ['background-color: #fff3cd' if v == '⬅ dipilih' else '' for v in col],
            subset=['Ket.']
        ),
        use_container_width=True, hide_index=True
    )
    st.caption("Baris yang disorot kuning = nilai K yang sedang dipilih pada slider.")

# ════════════════════════════════════════════════════════════════
# BAGIAN 3 — HASIL KOMPRESI: MODE GRAYSCALE
# ════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">🌑 Tahap 3 — Hasil Kompresi: Mode Grayscale (Luminansi)</div>', unsafe_allow_html=True)
st.caption(
    "Gambar dikonversi ke luminansi menggunakan bobot **ITU-R BT.601** "
    "(`0.2126R + 0.7152G + 0.0722B`) terlebih dahulu — mencerminkan sensitivitas mata manusia terhadap warna. "
    "Kemudian PCA dijalankan **1× pada matriks luminansi tersebut**. "
    "Metrik dihitung terhadap gambar asli versi grayscale."
)

tampilkan_section_hasil(
    arr_asli_visual=arr_gray_asli,
    arr_kompresi=arr_g,
    eigenvalues=ev_g,
    k=k, mse=mse_g, psnr=psnr_g, ssim_v=ssim_g,
    var=var_g, rasio=rasio,
    label_mode="Grayscale (Luminansi)",
    fn_histogram=plot_histogram_gray,
)

# Tabel benchmark grayscale
with st.expander("📋 Tabel Benchmark Multi-K — Mode Grayscale", expanded=True):
    with st.spinner("Menghitung benchmark grayscale..."):
        rows_g = benchmark_grayscale(arr_bytes, arr.shape, k)
    df_g = pd.DataFrame(rows_g)
    st.dataframe(
        df_g.style.apply(
            lambda col: ['background-color: #fff3cd' if v == '⬅ dipilih' else '' for v in col],
            subset=['Ket.']
        ),
        use_container_width=True, hide_index=True
    )
    st.caption("Baris yang disorot kuning = nilai K yang sedang dipilih pada slider.")

# ════════════════════════════════════════════════════════════════
# BAGIAN 4 — EDA SESUDAH KOMPRESI + PERBANDINGAN WARNA vs GRAYSCALE
# ════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">⚖️ Tahap 4 — EDA Sesudah Kompresi: Perbandingan Warna vs Grayscale</div>', unsafe_allow_html=True)
st.caption(
    "Perbandingan metrik antara dua mode. "
    "**Warna**: dibandingkan terhadap gambar asli RGB. "
    "**Grayscale**: dibandingkan terhadap gambar asli versi luminansi."
)

# Tabel perbandingan
comp = {
    'Metrik':        ['EVR (%)', 'MSE', 'PSNR (dB)', 'SSIM', 'Rasio Memori (%)'],
    'Mode Warna':    [f"{var_c*100:.2f}", f"{mse_c:.3f}", f"{psnr_c:.2f}", f"{ssim_c:.4f}", f"{rasio:.1f}"],
    'Mode Grayscale':[f"{var_g*100:.2f}", f"{mse_g:.3f}", f"{psnr_g:.2f}", f"{ssim_g:.4f}", f"{rasio:.1f}"],
}
st.dataframe(pd.DataFrame(comp), use_container_width=True, hide_index=True)

# Side-by-side gambar akhir
sc1, sc2, sc3 = st.columns(3)
with sc1:
    st.image(arr,      caption="Asli (RGB)",                width="stretch")
with sc2:
    st.image(arr_c,    caption=f"Warna K={k}", width="stretch")
with sc3:
    st.image(arr_g,    caption=f"Grayscale K={k}", width="stretch")

# ── PSNR interpretasi ──────────────────────────────────────────
def label_psnr(p):
    if p == float('inf'): return "SEMPURNA ★★★★★"
    if p >= 40: return "Sangat Baik ★★★★★"
    if p >= 35: return "Baik ★★★★"
    if p >= 28: return "Cukup ★★★"
    if p >= 20: return "Kurang ★★"
    return "Rendah ★"

st.info(
    f"**Kualitas Warna:** PSNR = {psnr_c:.2f} dB → {label_psnr(psnr_c)} | SSIM = {ssim_c:.4f}  \n"
    f"**Kualitas Grayscale:** PSNR = {psnr_g:.2f} dB → {label_psnr(psnr_g)} | SSIM = {ssim_g:.4f}"
)

# ── Footer ─────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "🏰 PCA Image Compressor | NumPy (PCA Manual) + scikit-image (SSIM) + Matplotlib  \n"
    "Rumus: Z = Xc·W (proyeksi) | X̂ = Z·Wᵀ + X̄ (rekonstruksi) | "
    "Rasio = k(H+W)×3 / H·W·3 × 100%"
)
