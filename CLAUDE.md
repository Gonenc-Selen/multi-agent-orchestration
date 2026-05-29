# CLAUDE.md — Agentic Energy CPR

> Bu dosya Claude Code için projenin "anayasası"dır. Her oturumda
> okunmalı; kod yazarken bu kurallara uyulmalıdır. Yeni bir karar
> alındığında "Karar Geçmişi" bölümüne eklenir.

---

## 1. Proje Özeti

**Proje adı:** Agentic Energy CPR (Common-Pool Resource)

**Amaç:** 3 hanenin paylaştığı bir ortak enerji kaynağı (şebeke
kapasitesi) üzerinde, her hane bir LLM ajanı tarafından temsil edilerek,
Common-Pool Resource oyunu oynanır. Sistem, ajanların müzakere etmeden
sadece niyet beyan ettikleri (V1) bir CPR davranışını test eder.

**Teorik temel:** Elinor Ostrom — Tragedy of the Commons (1990), tekrarlı
oyunlar, dağıtık karar verme.

**Veri kaynağı:** Mendeley Data — "Electrical demand/consumption and
climatological data of a small Mexican community" (DOI:
10.17632/vsjtbzjttb.4). 3 hane (H1, H6, H8) seçilmiştir.

**Tezin akademik konumu:** LLM-tabanlı ajanların gerçek tüketim verisi
üzerinde CPR davranışı sergileyip sergilemediğinin gözlemlenmesi.

---

## 2. Mutlak Kurallar (Bunları İhlal Etme)

Bu kurallar tasarım kararlarıdır, performans veya konfor için
değiştirilemez. Değişiklik gerekirse önce kullanıcıya sor.

1. **Makine öğrenmesi modeli eğitilmeyecek.** LSTM, ARIMA, Prophet,
   sklearn modelleri vs. KESİNLİKLE yok. Tahmin gereken her yerde ya
   fiziksel formül ya sabit varsayım kullanılır. LLM ajanın iç akıl
   yürütmesi bu kuralın istisnasıdır.

2. **LangGraph veya diğer ajan framework'leri kullanılmayacak.** CrewAI,
   AutoGen, LangChain ajan abstraksiyonları yasak. Custom sade Python
   sınıfları yazılır. `langchain-google-genai` gibi sadece model çağrısı
   için olan paketler OK.

3. **Çekirdek senaryodan bağımsız kalır.** `core/` klasöründeki hiçbir
   dosya "Puebla", "hane", "kWh" kelimesini sabit kodlamaz. Senaryoya
   özgü her şey `scenarios/*.yaml` dosyalarında yaşar.

4. **Her LLM çağrısı structured output döndürür.** Pydantic schema ile
   doğrulanır. Serbest metin parse etme yok.

5. **Random seed sabittir** (varsayılan 42). Tekrarlanabilirlik kritik.

6. **Veri belleğe bir kez yüklenir.** Her tur diskten okumak yasak.

7. **Tek bir hane = tek bir ajan.** Hane içine rol-bazlı alt-ajanlar
   eklenmez (V1).

---

## 3. Teknoloji Yığını

### Çalışma ortamı
- **Python:** 3.12
- **Paket yöneticisi:** `uv` (pip değil, poetry değil)
- **Virtual environment:** `uv venv` ile oluşturulur

### Çekirdek bağımlılıklar
- `pydantic` — veri modelleri ve schema doğrulama
- `pydantic-settings` — env tabanlı konfigürasyon
- `pyyaml` — senaryo dosyaları
- `pandas` — veri yükleme ve agregasyon
- `numpy` — sayısal işlemler
- `google-genai` — Vertex AI üzerinden Gemini çağrısı (resmi SDK)
- `google-cloud-aiplatform` — Vertex AI altyapı bağımlılığı
- `jinja2` — prompt şablonları
- `python-dotenv` — `.env` dosyası yükleme

### Örnek `pyproject.toml` bağımlılık bölümü
```toml
[project]
name = "agentic-energy"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "pydantic>=2.9",
    "pydantic-settings>=2.6",
    "pyyaml>=6.0",
    "pandas>=2.2",
    "numpy>=2.0",
    "google-genai>=0.3",
    "google-cloud-aiplatform>=1.70",
    "jinja2>=3.1",
    "python-dotenv>=1.0",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-mock>=3.14",
    "ruff>=0.7",
    "mypy>=1.13",
]
```

### Geliştirme bağımlılıkları
- `pytest` — test runner
- `pytest-mock` — LLM çağrı mock'lama
- `ruff` — linter + formatter (black kullanma, ruff yeterli)
- `mypy` — type checking

### LLM sağlayıcı
- **Platform:** Google Cloud Vertex AI (AI Studio değil)
- **Model:** `gemini-2.5-pro` (varsayılan)
- **SDK:** `google-genai` (yeni unified SDK, Aralık 2024+)
- **Kimlik doğrulama:** Application Default Credentials (ADC)
  - Geliştirici makinesinde: `gcloud auth application-default login`
  - CI/CD'de: Service Account JSON (gelecekte)
- **Temperature:** 0.6
- **Yedek model:** `gemini-2.5-flash` (test için, hız önemli olduğunda)

### Gerekli env değişkenleri (`.env` dosyasından okunur)
```
GOOGLE_CLOUD_PROJECT=<proje-id>
GOOGLE_CLOUD_LOCATION=us-central1
GEMINI_MODEL=gemini-2.5-pro
```

### Çağrı parametreleri ve örnek
```python
from google import genai
from google.genai import types
from core.schemas import AgentAction  # Pydantic modeli

client = genai.Client(
    vertexai=True,
    project=settings.google_cloud_project,
    location=settings.google_cloud_location,
)

response = client.models.generate_content(
    model=settings.gemini_model,
    contents=prompt,
    config=types.GenerateContentConfig(
        temperature=0.6,
        response_mime_type="application/json",
        response_schema=AgentAction,
        max_output_tokens=1024,
        seed=42,  # tekrarlanabilirlik
    ),
)
action: AgentAction = response.parsed
```

**Notlar:**
- `vertexai=True` parametresi kritik — bu olmadan AI Studio'ya bağlanır
- `response_schema` doğrudan Pydantic modeli alır — manual JSON parse yok
- Client tek seferde oluşturulur, çağrılarda yeniden kurulmaz (singleton)

---

## 4. Klasör Yapısı

```
agentic-energy/
├── CLAUDE.md                  # Bu dosya — anayasa
├── README.md                  # İnsan dostu giriş
├── pyproject.toml             # uv proje tanımı
├── .python-version            # 3.12
├── .env.example               # API key şablonu (.env GİT'e GİRMEZ)
├── .gitignore
├── scripts/
│   ├── setup.sh               # uv venv + uv sync + .env oluştur
│   ├── teardown.sh            # venv ve cache temizleme
│   └── prepare_data.py        # CSV temizleme + window kesme
├── core/                      # Senaryodan bağımsız çekirdek
│   ├── __init__.py
│   ├── agent.py               # HouseholdAgent sınıfı
│   ├── referee.py             # CPR ceza dağıtımı (deterministik)
│   ├── round_engine.py        # Tur döngüsü orkestratörü
│   ├── memory.py              # Hafıza yöneticisi (K=3 tur özet)
│   ├── llm_client.py          # Vertex AI Gemini sarmalayıcı (google-genai)
│   ├── schemas.py             # Pydantic modelleri
│   ├── pv_model.py            # PV fiziksel formülü
│   ├── metrics.py             # KPI hesaplayıcılar
│   ├── logger.py              # JSONL + CSV loglama
│   ├── config.py              # pydantic-settings
│   └── run.py                 # CLI giriş noktası
├── scenarios/                 # Swap edilebilir senaryolar
│   ├── puebla_yaz.yaml
│   ├── puebla_ilkbahar.yaml
│   └── puebla_kis.yaml
├── prompts/                   # Jinja2 şablonları
│   ├── system.j2
│   ├── decision.j2
│   └── memory_summary.j2
├── data/
│   ├── raw/                   # Orijinal CSV'ler (Mendeley)
│   │   ├── houseteh1_1day.csv
│   │   ├── houseteh6_1day.csv
│   │   ├── houseteh8_1day.csv
│   │   └── weather_outdoor.csv
│   └── processed/
│       └── community_window.csv  # H1+H6+H8 + PV + tarih, hazır
├── runs/                      # Her koşunun çıktısı (GİT'e GİRMEZ)
│   └── <timestamp>/
│       ├── config.yaml
│       ├── log.jsonl
│       ├── results.csv
│       └── metrics.json
├── tests/
│   ├── __init__.py
│   ├── unit/                  # Mock'lı, hızlı
│   │   ├── test_referee.py
│   │   ├── test_pv_model.py
│   │   ├── test_memory.py
│   │   └── test_schemas.py
│   ├── integration/           # Gerçek API, yavaş
│   │   └── test_smoke.py
│   └── conftest.py
└── analysis/                  # Notebook'lar, opsiyonel
    └── .gitkeep
```

---

## 5. Mimari Kurallar

### 5.1 Üç katman ayrımı
- **`core/`** — senaryodan bağımsız, değişmez. Sadece soyut "ajan",
  "tur", "hakem", "kaynak" kavramlarını bilir.
- **`scenarios/*.yaml`** — senaryoya özgü her şey. Persona'lar,
  kapasiteler, tarih aralıkları, prompt şablonları.
- **`data/`** — gerçek CSV verisi.

Bir şeyi nereye koyacağınızdan emin değilseniz: "Bu şey başka bir
senaryoda da var mı?" diye sorun. Evet ise `core/`, hayır ise `scenarios/`.

### 5.2 Bağımlılık yönü
- `core/` → hiçbir şeye bağlı değil (sadece stdlib + temel paketler)
- `scenarios/` → `core/`'u kullanır
- `data/` → kimseye bağlı değil, sadece veri

Ters bağımlılık YASAK. `core/` içinden `scenarios/` veya `data/` import
edilmez. Bunlar runtime'da yüklenir.

### 5.3 Konfigürasyon stratejisi
- **`.env`** — sırlar (API key'ler, sadece geliştirici makinesinde)
- **`scenarios/*.yaml`** — deney parametreleri (versiyonlanır)
- **`core/config.py`** — pydantic-settings, env'den okur, type-safe sağlar

### 5.4 Hata yönetimi
- LLM çağrısı 3 retry, exponential backoff
- Schema doğrulama başarısız olursa retry (LLM yeniden çağrılır)
- 3 retry'dan sonra hâlâ başarısız → tur başarısız sayılır, log'a yazılır,
  ajan o tur için "varsayılan eylem" alır (draw=kendi tüketimi, offer=0,
  store=0)

---

## 6. CPR Oyunu Kuralları

### 6.1 Genel mekanik
- N=3 ajan (H1, H6, H8)
- T=10 tur (1 tur = 1 gün)
- Ortak kaynak: günlük şebeke kapasitesi (C kWh)
- Her tur paralel niyet beyanı (V1'de pazarlık yok)
- Hakem deterministik, formül tabanlı (LLM değil)

### 6.2 Ajan eylemleri (her tur)
```json
{
  "draw_kwh": float,    // şebekeden çekilen (≥0)
  "offer_kwh": float,   // topluluğa verilen fazlalık (≥0)
  "store_kwh": float,   // bataryaya konulan (≥0, kapasite limitli)
  "reasoning": string   // 1-3 cümle açıklama (her zaman dolu)
}
```

### 6.3 Hakem mantığı (yumuşak orantılı ceza)
```
toplam_cekis = sum(agent.draw_kwh for agent in agents)
asim = max(0, toplam_cekis - kapasite)

for agent in agents:
    pay_orani = agent.draw_kwh / toplam_cekis
    brut_kazanc = agent.draw_kwh * 1.0  # birim fayda
    offer_kazanci = agent.offer_kwh * 0.5  # paylaşım ödülü
    ceza = (asim * pay_orani) * 3.0  # ceza katsayısı
    net = brut_kazanc + offer_kazanci - ceza
```

Parametreler (`scenario.yaml`'de override edilebilir):
- `unit_utility = 1.0`
- `share_bonus = 0.5`
- `penalty_multiplier = 3.0`

### 6.4 Belirsizlik tasarımı
- **Bugün:** ajan kendi tüketimini ve PV üretimini KESİN bilir
- **Yarın+:** ajan sadece mevsim genel bilgisini bilir
  ("yaz: yüksek güneş", "kış: az güneş"). Sayısal değer YOK.
- **Komşular:** ajan komşusunun **niyetini** görür ama bunun sözünün
  tutulup tutulmayacağını bilmez (V1'de pazarlık yok, niyet = eylem,
  ama yine de gözlem değil beyan üzerinden çalışır)

### 6.5 Batarya modeli
- Her ajan için sabit kapasite (yaml'de tanımlı)
- Verim %90 (doldur ve boşaltta toplam %10 kayıp)
- State of charge başlangıçta %50
- `store_kwh > batarya_kapasitesi - mevcut_doluluk` → kırpılır + log
- V1'de bataryadan çekiş otomatik (eksik kalan tüketim önce bataryadan
  karşılanır, sonra şebekeden)

---

## 7. Hafıza

### 7.1 Hafıza yapısı (her ajan için)
- **K=3 son tur** — daha eski turlar düşer
- **Kendi geçmişi:** her tur için (niyet, sonuç, kazanç) tuple'ı
- **Komşu gözlemleri:** her komşu için doğal dil özet
  - "H8 son 3 turun 2'sinde söz tuttu, 1'inde fazla çekti"
  - "H6 hep dengeli oynuyor"

### 7.2 Özet üretimi
- Her tur sonunda hafıza güncellenir
- Eski turlar bir cümleye sıkıştırılır (LLM çağrısı YAPILMAZ — şablon
  tabanlı, deterministik)
- Token bütçesi: her ajanın hafıza özeti ≤ 200 token

---

## 8. Veri Kontratı

### 8.1 Ham veri
- `data/raw/houseteh{1,6,8}_1day.csv` — sütunlar: `Time`, `kWh`
- `data/raw/weather_outdoor.csv` — sütunlar: `Time`, `Solar Radiation (W/m²)`, ...
- Eksik veri: boş hücre (NaN), `-1` değil

### 8.2 İşlenmiş veri (`prepare_data.py` üretir)
- `data/processed/community_window.csv` — tek dosya
- Sütunlar: `date`, `h1_kwh`, `h6_kwh`, `h8_kwh`, `solar_w_m2`,
  `h1_pv_kwh`, `h6_pv_kwh`, `h8_pv_kwh`
- 10 günlük pencere `scenario.yaml`'deki `start_date` ve `end_date`'e göre

### 8.3 Hane atamaları (V1)
| Hane | CSV | Profil | Persona | Panel m² |
|---|---|---|---|---|
| H1 | houseteh1_1day | Orta tüketici | "Geniş aile, klima yoğun" | 25 |
| H6 | houseteh6_1day | Küçük tüketici | "Dengeli, az tüketim" | 15 |
| H8 | houseteh8_1day | Büyük tüketici | "Yoğun, büyük panel" | 40 |

### 8.4 PV fiziksel formülü
```python
def daily_pv_kwh(radiation_w_m2: float, panel_area_m2: float,
                 efficiency: float = 0.18,
                 daylight_hours: float = 10.0) -> float:
    """Deterministik PV üretimi — ML değil."""
    return (radiation_w_m2 * panel_area_m2 * efficiency
            * daylight_hours / 1000.0)
```

### 8.5 Veri sınırı
Veri seti hanelerin tüm tüketimini değil belirli devreleri temsil
ediyor olabilir (ortalama 1.5-4 kWh/gün, gerçek hanelerden düşük).
Bu kabul edilmiştir, sentetik ölçeklendirme yapılmaz. Tezde dipnotta
belirtilir.

---

## 9. Kod Stili

### 9.1 Genel
- **Type hints zorunlu** — fonksiyon imzalarında, dönüş tiplerinde
- **mypy strict mode** çalışır olmalı
- **Docstring'ler Google stilinde**
- Maksimum satır uzunluğu: 100
- Tüm kod **senkron** (V1'de async yok)

### 9.2 Naming
- Sınıflar: `PascalCase` — `HouseholdAgent`, `Referee`
- Fonksiyonlar/değişkenler: `snake_case` — `compute_payoff`
- Sabitler: `UPPER_SNAKE` — `DEFAULT_TEMPERATURE`
- Private: tek alt çizgi prefix — `_internal_state`

### 9.3 Import organizasyonu
```python
# 1. stdlib
from datetime import date
from pathlib import Path

# 2. third-party
import pandas as pd
from pydantic import BaseModel

# 3. local
from core.schemas import AgentAction
```

### 9.4 Pydantic kullanımı
- Tüm "veri yapıları" Pydantic modeli olur (TypedDict veya dataclass değil)
- Validator'lar açık ve hatayı anlatan mesajlarla yazılır

### 9.5 Türkçe vs İngilizce
- **Kod (değişken, fonksiyon, sınıf isimleri):** İngilizce
- **Kod yorumları:** İngilizce
- **Docstring'ler:** İngilizce
- **Prompt şablonları (`prompts/*.j2`):** İngilizce
- **Persona açıklamaları (yaml içinde):** İngilizce
- **README, CLAUDE.md, tez:** Türkçe
- **Log mesajları:** İngilizce (machine-readable kalsın)

---

## 10. Test Stratejisi

### 10.1 Unit testler — `tests/unit/`
- **LLM çağrıları mock'lanır.** `pytest-mock` ile.
- Hızlı çalışır (< 5 saniye toplam).
- Her PR öncesi çalışır.
- Pure function'ları test eder: hakem mantığı, PV formülü, hafıza
  güncellemesi, schema doğrulama.

### 10.2 Integration / smoke testler — `tests/integration/`
- **Gerçek API'ya gider.** Manual olarak çalıştırılır
  (`pytest tests/integration/ --integration`).
- 3 ajan × 2 tur uçtan uca koşar.
- API key yoksa skip edilir.
- Toplam maliyet: ~$0.05.

### 10.3 Coverage hedefi
- Çekirdek (`core/`): %80+
- Prompt'lar: test edilmez (görsel kontrol)

---

## 11. Loglama ve Çıktı

### 11.1 Her koşu için
- **`runs/<timestamp>/config.yaml`** — kullanılan senaryo + git hash
- **`runs/<timestamp>/log.jsonl`** — her satır bir event (LLM çağrısı,
  hakem kararı, hafıza güncelleme)
- **`runs/<timestamp>/results.csv`** — tur bazında özet tablo
- **`runs/<timestamp>/metrics.json`** — toplam KPI'lar

### 11.2 İzlenecek KPI'lar
**Topluluk:**
- `total_welfare` — toplam net kazanç
- `capacity_violation_count` — kapasite ihlali olan tur sayısı
- `capacity_violation_avg_kwh` — ortalama ihlal miktarı
- `gini_coefficient` — kazanç eşitsizliği
- `self_sufficiency_ratio` — topluluk içi karşılanan tüketim oranı

**Bireysel (her ajan için):**
- `net_profit` — net kazanç
- `total_offered_kwh` — paylaşım katkısı
- `violation_contribution_kwh` — ihlale katkı

### 11.3 Token izleme
- Her LLM çağrısının prompt + completion token sayısı log'a yazılır
- Koşu sonunda toplam token + tahmini USD maliyet raporlanır

---

## 12. Çalıştırma

### 12.1 İlk kurulum

**Ön gereksinim:** Google Cloud Console'da bir proje oluşturulmuş ve
Vertex AI API enable edilmiş olmalı. Ayrıca makinede `gcloud` CLI
kurulu olmalı.

```bash
# 1. Bağımlılıkları kur
./scripts/setup.sh
# uv venv oluşturur, uv sync ile bağımlılıkları kurar,
# .env.example dosyasını .env olarak kopyalar

# 2. .env dosyasını doldur
# Editör ile aç ve şu değişkenleri doldur:
#   GOOGLE_CLOUD_PROJECT=<proje-id>
#   GOOGLE_CLOUD_LOCATION=us-central1
#   GEMINI_MODEL=gemini-2.5-pro

# 3. Google Cloud'a giriş yap (bir kerelik)
gcloud auth application-default login
gcloud config set project <proje-id>

# 4. Vertex AI API'sinin enable olduğunu doğrula
gcloud services enable aiplatform.googleapis.com

# 5. Bağlantı testi (smoke test)
uv run python -m core.run --scenario scenarios/puebla_yaz.yaml --runs 1 --smoke
```

### 12.2 Veri hazırlama
```bash
uv run python scripts/prepare_data.py \
  --scenario scenarios/puebla_yaz.yaml
# community_window.csv üretir
```

### 12.3 Deney çalıştırma
```bash
# Tek koşu
uv run python -m core.run --scenario scenarios/puebla_yaz.yaml

# Çok koşu (istatistik için)
uv run python -m core.run --scenario scenarios/puebla_yaz.yaml --runs 5

# Smoke test
uv run python -m core.run --scenario scenarios/puebla_yaz.yaml --runs 1 --smoke
```

### 12.4 Test
```bash
# Unit testler (hızlı, mock)
uv run pytest tests/unit/

# Integration testler (gerçek API)
uv run pytest tests/integration/ --integration

# Lint + type check
uv run ruff check .
uv run mypy core/
```

### 12.5 Temizleme
```bash
./scripts/teardown.sh
# .venv, runs/, __pycache__ temizler
```

---

## 13. .gitignore Gerekenleri

```
.venv/
__pycache__/
*.pyc
.env
runs/
data/processed/
.pytest_cache/
.mypy_cache/
.ruff_cache/
*.egg-info/
dist/
build/
.DS_Store
.idea/
.vscode/settings.json
```

---

## 14. Karar Geçmişi

Bu bölüm projenin "neden böyle?" sorusunun cevabıdır. Her büyük karar
buraya yazılır, gerekçesiyle.

### KG-001: ML yok
**Tarih:** İlk planlama  
**Karar:** Klasik ML modeli (LSTM, ARIMA, sklearn) eğitilmeyecek.  
**Gerekçe:** Tezin asıl katkısı agentic davranıştır, tahmin doğruluğu
değil. Tahmin gerekenler ya fiziksel formül ya sabit varsayım. ML eklemek
kapsamı genişletir, vakit alır, tezin odağını dağıtır.

### KG-002: LangGraph kullanılmıyor
**Tarih:** İlk planlama  
**Karar:** Custom sade Python sınıflarıyla yazılacak.  
**Gerekçe:** CPR oyunu LangGraph'ın sweet spot'u değil. State şeması katı,
ajan iç mantığı zorla framework'e oturtuluyor. Custom yaklaşım daha sade,
daha kontrollü, debug edilebilir. Çekirdek ~400 satır tahmini.

### KG-003: Gemini 2.5 Pro üzerinden Vertex AI
**Tarih:** İlk planlama  
**Karar:** Google Cloud Vertex AI üzerinden `gemini-2.5-pro` kullanılır.
AI Studio değil. Project ID + Location bazlı bağlantı yapılır.  
**Gerekçe:** Kullanıcı Google Cloud'da ücretsiz kredi sahibi. Pro modeli
müzakere/akıl yürütme için Flash'tan belirgin üstün. Vertex AI kurumsal
düzeyde kimlik doğrulama ve faturalandırma sağlar.

### KG-004: Hakem deterministik
**Tarih:** İlk planlama  
**Karar:** Hakem LLM çağrısı yapmaz, formül tabanlı çalışır.  
**Gerekçe:** Token tasarrufu, %100 tekrarlanabilirlik, klasik CPR oyunu
tanımına uygun. "Hakem ajanı" sadece konsept olarak ajan, teknik olarak
saf koddur.

### KG-005: Yumuşak orantılı ceza
**Tarih:** İlk planlama  
**Karar:** Aşım durumunda her ajan çekiş payına göre ceza yer
(katsayı=3).  
**Gerekçe:** Sert eşik (herkes 0) çok cezalandırıcı, asimetrik
hesaplaması karmaşık. Yumuşak orantılı hem Ostrom literatüründe yaygın
hem matematiksel olarak temiz.

### KG-006: Pazarlık ve koalisyon V1'de yok
**Tarih:** İlk planlama  
**Karar:** V1 sadece niyet beyanı içerir. Pazarlık ve koalisyon V2+.  
**Gerekçe:** "Scaffold-first" prensibi. Çekirdek doğru çalıştığı
doğrulandıktan sonra zenginleştirilir. Her özellik ayrı ablation
çalışması olarak eklenir.

### KG-007: Senkron çalışma
**Tarih:** İlk planlama  
**Karar:** Tüm çekirdek senkron. Async yok.  
**Gerekçe:** Debug kolaylığı. 3 ajan × 10 tur × 5 koşu = 150 LLM çağrısı,
async olsa bile ~1 dakika fark eder. Bütçeli geliştirme için senkron
yeterli.

### KG-008: 3 hane (H1, H6, H8)
**Tarih:** İlk planlama  
**Karar:** Mendeley veri setinden H1, H6, H8 seçilir.  
**Gerekçe:** Üçü de 377/377 gün eksiksiz. Tüketim hiyerarşisi net:
H6 küçük (1.56), H1 orta (3.90), H8 büyük (4.29). H7 yarı eksik (atılır),
H10 aykırı değerli (atılır).

### KG-009: Her ev = bir ajan (rol-bazlı değil)
**Tarih:** İlk planlama  
**Karar:** Smart House Operator'ın rol-bazlı (AC ajanı, Işık ajanı vs)
yapısı kullanılmaz. Bir hane bir LLM ajanıdır.  
**Gerekçe:** CPR oyunu ekonomik aktör seviyesinde stratejik etkileşim
gerektirir. Rol-bazlı yapı işbirliği için tasarlanır, çelişki için değil.

### KG-010: Reasoning her zaman açık
**Tarih:** İlk planlama  
**Karar:** Ajan JSON çıktısında `reasoning` alanı her tur dolu.  
**Gerekçe:** Tezin nitel analiz bölümü için kritik. Token maliyeti
ihmal edilebilir (~50 token/tur/ajan).

### KG-011: Belirsizlik sadece sosyal
**Tarih:** İlk planlama  
**Karar:** Fiziksel belirsizlik yok. Bugün için kesin veri, yarın için
sadece mevsim bilgisi. Tüm gerilim komşu davranışından gelir.  
**Gerekçe:** Gürültü ekleme kodu yok → daha az parametre. Tezin odağı
sosyal etkileşim, fiziksel tahmin değil.

### KG-012: LiteLLM yerine google-genai SDK
**Tarih:** İlk planlama (revize)  
**Karar:** Çağrı sarmalayıcı kütüphane (LiteLLM, LangChain wrapper)
kullanılmaz. Doğrudan `google-genai` resmi SDK'sı ile Vertex AI'ya
bağlanılır.  
**Gerekçe:** Tek bir sağlayıcı kullanılacak (Gemini), soyutlama
katmanına gerek yok. Vertex AI'nin `response_schema` desteği LiteLLM'in
JSON wrapper'ından daha sağlam. Daha az bağımlılık, daha açık hata
mesajları.

### KG-013: Application Default Credentials ile auth
**Tarih:** İlk planlama (revize)  
**Karar:** Vertex AI kimlik doğrulama için ADC kullanılır. Service
account JSON key'i geliştirme makinesinde tutulmaz.  
**Gerekçe:** `gcloud auth application-default login` ile tek seferde
giriş yeterli. JSON key dosyası saklama, yanlışlıkla commit'leme riski
ortadan kalkar. CI/CD'de gerekirse Service Account'a geçilir.

### KG-014: Kıtlık rejimi kalibrasyonu
**Tarih:** 2026-05-28  
**Karar:** Panel boyutları gerçek fiziksel değerlerden (~25/15/40 m²)
çok daha küçük değerlere indirildi (H1=5, H6=3, H8=8 m²).
`grid_capacity_kwh` 20.0'dan 5.5'e düşürüldü.  
**Gerekçe:** Gerçek veri üzerinde yapılan ön analiz, PV/tüketim oranının
~2-3x çıktığını gösterdi. Bu durumda ajanlar şebekeden neredeyse hiç
çekmediği için CPR ikilemini (kıtlık → aşırı çekim → ceza) hiç
yaşamadı; 10 turun tamamında ihlal sıfır kaldı. Hedef: 10 turun 3-5'inde
kapasite ihlali oluşturarak gerçek bir ortak-havuz gerilimi yaratmak.
Senaryo dosyalarındaki bu parametreler "fiziksel gerçeklik" değil
"oyun dengesi" için ayarlanmıştır — tezde dipnotta belirtilecek.


KG-015: YAML doğrulama zorunluluğu
Tüm scenario dosyaları, çalıştırılmadan önce Pydantic ScenarioConfig modeli ile doğrulanır. Ham dict[str, Any] döndürme kabul edilmez. Yeni alan eklendiğinde önce schema güncellenir.


---

## 15. Sıkça Karıştırılan Noktalar (Claude Code İçin)

**"Tahmin lazım, ML kuralım mı?"**
HAYIR. Fiziksel formül veya sabit varsayım kullan. ML yasak (KG-001).

**"Ajan paralel mi karar versin?"**
HAYIR. V1'de senkron, sırayla. asyncio yok (KG-007).

**"LangChain araç ekleyeyim mi?"**
HAYIR. `google-genai` zaten çağrı yapıyor. Ajan framework'ü yok (KG-002).

**"Hakem de LLM mi olsun?"**
HAYIR. Deterministik formül (KG-004).

**"Pazarlık ekleyelim, daha zengin olur"**
HAYIR. V1 kapalı. V2'ye yaz, şimdi ekleme (KG-006).

**"5 hane kullanalım, veri daha zengin"**
HAYIR. 3 hane (KG-008). Daha fazlası V2.

**"Bütün veriyi belleğe yüklemek savurganlık değil mi?"**
HAYIR. 3 hane × 377 gün = trivial boyut. Cache şart (Kural 6).

**"LiteLLM kullanalım, sağlayıcı bağımsız olur"**
HAYIR. Direkt `google-genai` (KG-012). Tek sağlayıcı var (Gemini),
soyutlamaya gerek yok.

**"AI Studio API key ile bağlanalım, daha kolay"**
HAYIR. Vertex AI üzerinden Project ID + Location ile bağlanılır
(KG-003). `vertexai=True` parametresi zorunlu.

**"GOOGLE_APPLICATION_CREDENTIALS dosya yolu nereye yazılacak?"**
Hiçbir yere. ADC kullanıyoruz (KG-013). `gcloud auth application-default
login` ile giriş yapılır, SDK otomatik bulur.

---

## 16. Bir Soruyla Karşılaşırsanız

Eğer Claude Code bir tasarım kararı verirken emin değilse:

1. Önce bu CLAUDE.md'yi tekrar oku
2. Karar Geçmişi (Bölüm 14) bölümünde benzer karar var mı bak
3. Hâlâ belirsizse: KULLANICIYA SOR. Sessizce bir yön seçme.

Geri dönüşü olmayan veya bütçeyi etkileyen kararlarda her zaman onay al.
