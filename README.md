# 🎧 Content-Aware RecSys: 4 Signals Analysis

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Implicit](https://img.shields.io/badge/Implicit-0.7.3-orange?style=flat)](https://github.com/benfred/implicit)
[![Polars](https://img.shields.io/badge/Polars-1.42.1-blue?style=flat&logo=polars&logoColor=white)](https://pola.rs/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3.0-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Gensim](https://img.shields.io/badge/Gensim-4.3.0-green?style=flat)](https://radimrehurek.com/gensim/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Hub-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Исследование влияния контентных эмбеддингов на качество рекомендаций.** Сравнительный анализ 4 типов контентных сигналов (текстовый, аудио, графовый, мультимодальный) и их линейное смешивание с коллаборативной фильтрацией (Implicit ALS) на датасетах nowplaying-RS и VK-LSVD.

---

## ✨ Возможности

| Иконка | Возможность | Описание |
| :---: | :--- | :--- |
| 📊 | **Агрегация неявных сигналов** | Взвешивание взаимодействий через $\log(1 + x)$ и действие-специфичные бонусы |
| 🧬 | **4 типа контентных сигналов** | Текстовые (TF-IDF+SVD), аудио (Spotify), граф (Item2Vec), мультимодальные (VK) |
| ⚖️ | **Гибридные модели** | Линейное смешивание скоров CF и контентной модели с подбором коэффициента $\alpha$ |
| ⏱️ | **Временная валидация** | Оценка по протоколу *Temporal Split* с расчетом метрик Recall@10 и NDCG@10 |
| 🧊 | **Анализ Cold-Start** | Оценка качества ранжирования на сегменте малопопулярных объектов ("длинный хвост") |
| 🔬 | **Permutation Importance** | Оценка информационной важности модальностей через градиентный бустинг |

---

## 🗂 Структура проекта

```text
content-aware-recsys/
├── data/
│   ├── nowplaying_rs_dataset/          # 🎵 Датасет nowplaying-RS (Zenodo)
│   └── VK-LSVD/                        # 🎬 Датасет мультимодальных эмбеддингов VK
├── four_signals_analysis_FINAL.ipynb   # 📓 Основной рабочий ноутбук с экспериментами
├── requirements.txt                    # 📦 Список зависимостей
└── README.md
