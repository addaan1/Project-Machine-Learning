# Deploy EcoDash ke Hugging Face Spaces

Dokumen ini adalah jalur deploy gratis untuk demo ujian final project. Target utama adalah Hugging Face Spaces dengan SDK `Docker`.

## 1. Buat Space

1. Buka https://huggingface.co/spaces.
2. Klik **Create new Space**.
3. Isi:
   - **Space name**: `ecodash-indonesia` atau nama lain yang tersedia.
   - **SDK**: `Docker`.
   - **Visibility**: `Public` untuk gratis.
4. Setelah Space dibuat, buka tab **Settings**.
5. Pastikan hardware tetap **CPU Basic**.

## 2. Set Environment Variables

Tambahkan variable berikut di **Settings > Variables and secrets**:

```text
DJANGO_ENV=production
DEBUG=False
ALLOWED_HOSTS=.hf.space,localhost,127.0.0.1,0.0.0.0
CSRF_TRUSTED_ORIGINS=https://*.hf.space
SECRET_KEY=<isi-dengan-random-secret-key>
```

Jika memakai script `scripts/deploy_hf_space.ps1`, variable production dasar dan `SECRET_KEY` akan dibuat otomatis saat deploy. Tetap cek tab **Settings > Variables and secrets** setelah upload untuk memastikan semua variable tersimpan.

Untuk membuat `SECRET_KEY` lokal:

```powershell
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

## 3. Upload File

Repo ini sudah berisi file deploy utama:

- `Dockerfile`
- `requirements-deploy.txt`
- `.dockerignore`
- `hf-space/README.md`

Jika upload lewat Hugging Face CLI, file `hf-space/README.md` perlu dikirim sebagai `README.md` di root Space supaya metadata Docker terbaca.

Contoh alur CLI manual:

```powershell
hf auth login
hf repos create <username>/ecodash-indonesia --type space --space-sdk docker --exist-ok
hf upload <username>/ecodash-indonesia . --type space --exclude ".git/*" --exclude "notebooks/*" --exclude "*.ipynb" --exclude "dashboard/staticfiles/*"
hf upload <username>/ecodash-indonesia hf-space/README.md README.md --type space
```

Jika upload manual lewat website, pastikan `README.md` di Space memakai isi dari `hf-space/README.md`.

Alternatif yang paling ringkas dari root repo:

```powershell
.\scripts\deploy_hf_space.ps1 -SpaceId "<username>/ecodash-indonesia"
```

## 4. Validasi Setelah Build

Setelah build selesai, buka:

```text
https://<username>-ecodash-indonesia.hf.space/
```

Cek halaman berikut:

- `/`
- `/dashboard/`
- `/forecasting/`
- `/daya-beli/`
- `/map/`
- `/api/usd-idr/`

Untuk presentasi, buka website 10-15 menit sebelum demo agar Space sudah aktif dan tidak cold start.

## Catatan

- Server tidak menjalankan retraining model.
- Data dan artefak model dibaca dari folder `datasets/` dan `models/`.
- SQLite cukup untuk demo karena aplikasi tidak menyimpan data pengguna.
- Docker lokal belum wajib; build resmi akan dijalankan oleh Hugging Face Spaces.
