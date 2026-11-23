---
name: nmap
description: TCP/UDP port ve servis tarayıcı; recon'un ilk adımı. Açık port ve servis/versiyon tespiti.
tool: nmap
---

## Kanonik kullanım
`arac`: `nmap`, `parametreler`: `{"hedef": "<ip/cidr/host>", "secenekler": "<flag'ler>"}`.
Tipik: `-sV -Pn -p-` (tüm portlar + servis/versiyon). Hedef parametreye ayrı yaz, `secenekler` içine gömme.

## Ana flag'ler
- `-sV` servis/versiyon, `--version-light` daha hızlı versiyon tespiti.
- `-p-` tüm 65535 port, `-p 80,443` seçili, `--top-ports 100` hızlı.
- `-Pn` ping'i atla (host ping'e cevap vermiyorsa şart), `-n` DNS çözme, `--open` yalnız açık portlar, `-v` ayrıntı.
- `-sT` TCP connect, `-sS` SYN (root gerekir), `-sU` UDP (yavaş).

## Policy'nin izin vermedikleri
Bu ortamda şunlar REDDEDİLİR, kullanma: NSE scriptleri (`-sC`, `--script`), `-A`, `-O`, `-T<n>`, dosya çıktısı (`-oN`/`-oX`), `secenekler` içinde ek hedef. Reddedilirse çağrıyı yalnız yukarıdaki flag'lerle tekrarla.

## Tuzaklar
- `-sS` root ister; yetki yoksa `-sT` (connect) kullan.
- `-p-` yavaş olabilir; hızlı ilk bakış için `--top-ports 100`, sonra açık portlarda `-sV -p <liste>`.
- `masscan` çok geniş aralıkta daha hızlı — sonra nmap ile derinleştir.

## Güvenlik/kapsam
Yalnızca yetkili/lab hedefi. Hedef IP/host kapsam (scope) içinde olmalı; policy scope dışını reddeder.
