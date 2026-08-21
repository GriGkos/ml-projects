# Кредитный скоринг: Home Credit Default Risk

Учебный ML-проект по прогнозированию вероятности дефолта клиента в соревновании [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/overview).

Проект выполнен как полное воспроизводимое исследование: от EDA и контроля качества данных до feature engineering, 5-fold OOF-валидации классических моделей, MLP, SHAP-интерпретации, fairness-проверки и итогового Kaggle submission.

## Что исследуется

- Объясняется качество исходных данных: уникальность заявок, дисбаланс классов, пропуски, аномальные значения и покрытие вспомогательных источников.
- Строятся признаки из анкеты, бюро, предыдущих заявок, POS/Cash, платежей и кредитных карт. Все таблицы агрегируются до `SK_ID_CURR` до соединения с заявками.
- Сравниваются одиночное дерево, логистическая регрессия, LightGBM, CatBoost и MLP. Основная метрика - ROC-AUC; дополнительная - PR-AUC, F1, Brier score и время обучения.
- LightGBM настраивается Optuna, финальный blend выбирается только по OOF-прогнозам, без подбора под публичный лидерборд.

Уже проверенный контрольный baseline LightGBM только по анкете получил **OOF ROC-AUC 0.76802**. В полном запуске этот результат служит точкой отсчёта для оценки добавочной ценности кредитной истории.

## Структура

```text
04_home_credit_scoring/
|-- data/raw/                     # исходные CSV Kaggle (в .gitignore)
|-- notebooks/home_credit_scoring.ipynb
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

## Методологические гарантии

- Одинаковые стратифицированные 5 фолдов для сравнения моделей.
- Все fit-преобразования выполняются только на train-части фолда.
- Контроль `one_to_one` при каждом соединении с агрегированными источниками.
- Нет target encoding или признаков, использующих `TARGET` вне training fold.
- SHAP и fairness интерпретируются как диагностика модели, а не причинное доказательство.
