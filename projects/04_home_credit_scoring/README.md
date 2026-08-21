# Кредитный скоринг: Home Credit Default Risk

Учебный ML-проект по прогнозированию вероятности дефолта клиента в соревновании [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/overview).

Проект выполнен как полное воспроизводимое исследование: от EDA и контроля качества данных до feature engineering, 5-fold OOF-валидации классических моделей, MLP, SHAP-интерпретации, fairness-проверки и итогового Kaggle submission.

## Что исследуется

- Объясняется качество исходных данных: уникальность заявок, дисбаланс классов, пропуски, аномальные значения и покрытие вспомогательных источников.
- Строятся признаки из анкеты, бюро, предыдущих заявок, POS/Cash, платежей и кредитных карт. Все таблицы агрегируются до `SK_ID_CURR` до соединения с заявками.
- Сравниваются одиночное дерево, логистическая регрессия, LightGBM, CatBoost и MLP. Основная метрика - ROC-AUC; дополнительная - PR-AUC, F1, Brier score и время обучения.
- LightGBM настраивается Optuna, финальный blend выбирается только по OOF-прогнозам, без подбора под публичный лидерборд.

Контрольный baseline и полный ансамбль намеренно пересчитываются одним и тем же fold-safe протоколом перед финальным отчётом. В README и PDF попадут только результаты этого чистого запуска, а не метрики из ранних версий feature pipeline.

## Структура

```text
04_home_credit_scoring/
|-- data/raw/                     # исходные CSV Kaggle (в .gitignore)
|-- notebooks/home_credit_scoring.ipynb
|-- scripts/run_full_gpu.ps1         # воспроизводимый полный GPU-запуск
|-- reports/figures/              # графики, созданные ноутбуком
|-- submissions/                  # финальный CSV для Kaggle
|-- README.md
`-- requirements.txt
```

## Данные

Нужны все CSV из архива соревнования:

- `application_train.csv`, `application_test.csv`;
- `bureau.csv`, `bureau_balance.csv`;
- `previous_application.csv`, `POS_CASH_balance.csv`;
- `installments_payments.csv`, `credit_card_balance.csv`.

Расположите их в `data/raw/`. Исходные данные и любые подготовленные матрицы намеренно не хранятся в Git: ноутбук каждый раз строит признаки в памяти из исходных файлов и не использует кэши.

## Запуск

```powershell
pip install -r projects/04_home_credit_scoring/requirements.txt
cd projects/04_home_credit_scoring
jupyter lab notebooks/home_credit_scoring.ipynb
```

В ноутбуке `FAST_MODE=False` - это финальный режим, который использует полный датасет. `FAST_MODE=True` предназначен только для smoke-теста логики на ограниченном числе строк и не должен использоваться для отчёта или submission.

Полный эксперимент с Optuna, 5-fold LightGBM/CatBoost и 5-fold MLP рекомендуется выполнять на машине с 16+ ГБ RAM; CUDA заметно ускоряет CatBoost и MLP. Никакие промежуточные признаки, модели или кэши при этом не записываются на диск.

### Финальный запуск на GPU

1. Откройте PowerShell в `projects/04_home_credit_scoring` и оставьте `FAST_MODE=False` (это режим по умолчанию).
2. Перед запуском установите `$env:HOME_CREDIT_RUN_FULL='1'`; для более тщательного поиска можно добавить `$env:HOME_CREDIT_OPTUNA_TRIALS='80'`.
3. Выполните ноутбук сверху вниз: EDA, построение признаков и leakage-audit должны завершиться до обучения моделей.

То же самое без ручной работы в Jupyter (и с HTML-копией выполненного отчёта):

```powershell
cd projects/04_home_credit_scoring
.\scripts\run_full_gpu.ps1 -OptunaTrials 80
```

Скрипт сознательно остановится, если CUDA недоступна: финальный режим не должен незаметно превратиться в многодневное CPU-обучение.

Финальная ячейка сама запускает 40 trials Optuna, 5-fold LightGBM, CatBoost и MLP, подбирает веса исключительно по OOF, создаёт `submissions/submission_oof_blend.csv`, а также строит SHAP и fairness-диагностику. Не нужно вручную раскомментировать отдельные фрагменты кода.

Smoke-проверка обновлённого feature engineering пройдена на реальных CSV: для всех пяти источников сохранён уникальный `SK_ID_CURR`, а числовые признаки не содержат бесконечностей. После добавления временных агрегатов на малом срезе бюро даёт 490 признаков, предыдущие заявки — 494; полная размерность намеренно фиксируется только результатом запуска на всех строках.

## Методологические гарантии

- Одинаковые стратифицированные 5 фолдов для сравнения моделей.
- Все fit-преобразования выполняются только на train-части фолда.
- Контроль `one_to_one` при каждом соединении с агрегированными источниками.
- Нет target encoding или признаков, использующих `TARGET` вне training fold.
- Leakage-audit выполняется в двух слоях: runtime-инварианты проверяют матрицу данных, а статический audit проверяет исходники feature builders на обращение к `TARGET`, target encoding, небезопасные merge и test-зависимое кодирование категорий.
- SHAP и fairness интерпретируются как диагностика модели, а не причинное доказательство.
