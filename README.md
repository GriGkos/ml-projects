# ML Projects

Коллекция проектов по машинному обучению: от базовых ML-пайплайнов до полноценных Kaggle-решений с feature engineering, out-of-fold валидацией, подбором гиперпараметров, интерпретацией моделей и внешней проверкой на leaderboard.

Основной акцент — не на количестве моделей, а на последовательной проверке гипотез, корректной валидации и понимании того, **за счёт чего меняется качество**.

## Featured projects

### 04. Home Credit Default Risk

**Кредитный скоринг по текущей заявке и подробной истории клиента.**

`CatBoost` · `LightGBM` · `PyTorch` · `Optuna` · `SHAP` · `Permutation Importance`

| Результат | Значение |
| --- | ---: |
| OOF ROC-AUC | **0.79405** |
| Kaggle Public | **0.79479** |
| Kaggle Private | **0.79510** |

Что интересно в проекте:

- отдельно измерен вклад нелинейной модели и вклад подробной кредитной истории;
- финальная матрица содержит **2 708 признаков** из текущей заявки и исторических таблиц;
- сравниваются Logistic Regression, Decision Tree, LightGBM, CatBoost и MLP;
- ансамбль CatBoost + LightGBM проверен через **второй уровень cross-validation**, а не на тех же OOF, по которым подбирались веса;
- group permutation importance показывает, какие источники истории реально дают сигнал;
- отдельно выполнены SHAP-анализ, разбор FP/FN, calibration и fairness-диагностика.

**Главный результат эксперимента:**

```text
Logistic Regression             0.75141
LightGBM — только заявка        0.76724
LightGBM — заявка + история     0.78538
CatBoost — заявка + история     0.79405
Cross-fitted blend              0.79440
```

История клиента дала LightGBM `+0.01814 ROC-AUC`, а среди исторических источников наиболее полезными оказались `bureau`, `installments` и `previous_application`.

[Открыть проект →](./projects/04_home_credit_scoring/)

---

### 03. Elo Merchant Category Recommendation

**Регрессия лояльности клиента по истории транзакций.**

`XGBoost` · `LightGBM` · `CatBoost` · `Optuna` · `Target Encoding` · `Bootstrap`

| Результат | Значение |
| --- | ---: |
| Private RMSE | **3.60826** |
| Public RMSE | **3.69615** |
| Private leaderboard | **82 / 4 111** |
| Место | **Top 1.99%** |

Что интересно в проекте:

- работа с historical transactions объёмом более **29 млн строк**;
- чтение больших таблиц чанками и агрегация до уровня карты;
- feature engineering по historical и new periods;
- fold-safe target encoding;
- подбор LightGBM, XGBoost и CatBoost через Optuna;
- seed bagging для XGBoost;
- weighted blend и ridge stacking;
- статистическая проверка небольшого преимущества ансамбля через **paired bootstrap**.

Формально лучший OOF показал stacking, но bootstrap не подтвердил устойчивое преимущество над более простым XGBoost seed bagging. Поэтому для Kaggle выбран более простой и стабильный вариант.

[Открыть проект →](./projects/03_elo_merchant/)

---

### 02. Online Shoppers Purchasing Intention

**Предсказание покупки по поведению пользователя в интернет-магазине.**

`scikit-learn` · `Pipeline` · `GridSearchCV` · `Feature Selection` · `Permutation Importance`

| Модель | ROC-AUC | Recall | Precision |
| --- | ---: | ---: | ---: |
| Gradient Boosting | **0.937** | 0.605 | **0.717** |
| Random Forest | 0.934 | 0.788 | 0.590 |
| Logistic Regression | 0.913 | **0.806** | 0.519 |

В проекте последовательно проверяются гипотезы о влиянии балансировки классов, масштабирования, категориальных признаков, feature selection и удаления отдельных признаков. Все preprocessing-шаги находятся внутри `Pipeline`, а финальный test используется только один раз после выбора моделей.

[Открыть проект →](./projects/02_online_shoppers/)

---

## ML foundations

### 01. ML Bricks

Базовые элементы корректного ML-пайплайна на `load_breast_cancer`:

- стратифицированный train/test split;
- импутация без утечки;
- One-Hot Encoding;
- расчёт метрик;
- baseline через `DummyClassifier`;
- cross-validation;
- `GridSearchCV`;
- feature importance дерева решений.

[Открыть проект →](./projects/01_ml_bricks/)

### 00. NumPy & pandas

Небольшой вводный проект по работе с табличными данными: загрузка Excel, анализ структуры `DataFrame`, фильтрация, группировки и обработка смешанных типов данных.

[Открыть проект →](./projects/00_numpy_pandas/)

---

## Как я строю эксперименты

Во всех крупных проектах использую примерно одну и ту же логику:

1. **Постановка вопроса.** Сначала формулируется, что именно нужно проверить, а не просто «обучить побольше моделей».
2. **EDA.** Проверяются распределения, пропуски, аномалии, баланс классов и структура связанных таблиц.
3. **Простой baseline.** Нужен понятный нижний уровень, относительно которого можно измерять дальнейший прирост.
4. **Fold-safe preprocessing.** Преобразования, использующие train-статистики или target, выполняются только внутри обучающей части соответствующего фолда.
5. **Одинаковая CV-схема.** Основные модели сравниваются на одних и тех же разбиениях.
6. **Анализ прироста.** Проверяется не только итоговая метрика, но и то, какие данные, признаки или решения реально дали улучшение.
7. **Интерпретация.** Используются feature importance, permutation importance, SHAP, error analysis и дополнительные диагностические метрики.
8. **Внешняя проверка.** Для Kaggle-проектов локальный OOF сопоставляется с public/private leaderboard.

---

## Стек

**Machine Learning**  
`scikit-learn` · `LightGBM` · `XGBoost` · `CatBoost` · `Optuna`

**Deep Learning**  
`PyTorch`

**Data & Analysis**  
`pandas` · `NumPy` · `SHAP` · `Matplotlib` · `Jupyter`

**Engineering**  
`Git` · `Docker` · `CUDA`

---

## Структура репозитория

Все проекты собраны в одной ветке и доступны как обычные директории:

```text
ml-projects/
├── projects/
│   ├── 00_numpy_pandas/
│   ├── 01_ml_bricks/
│   ├── 02_online_shoppers/
│   ├── 03_elo_merchant/
│   └── 04_home_credit_scoring/
├── README.md
├── requirements.txt
└── .gitignore
```

У каждого проекта есть собственный README с постановкой задачи, результатами, структурой файлов и инструкцией по воспроизведению.

Чтобы открыть репозиторий локально:

```bash
git clone https://github.com/GriGkos/ml-projects.git
cd ml-projects
```

После этого можно перейти в нужную директорию `projects/<project_name>/`.

Исходные Kaggle-данные, кэши и обученные модели в Git не добавляются.

---

## Коротко

Если хочется посмотреть только самые сильные работы, я бы начал с:

1. **Home Credit Default Risk** — кредитный скоринг, большая историческая матрица, CatBoost, SHAP, permutation importance и fairness;
2. **Elo Merchant Category Recommendation** — большие транзакционные таблицы, feature engineering, boosting и top 1.99% на Kaggle;
3. **Online Shoppers Purchasing Intention** — компактный и хорошо читаемый пример полного классификационного pipeline.
