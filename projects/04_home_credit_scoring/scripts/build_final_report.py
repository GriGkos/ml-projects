"""Собирает 5-страничный PDF-отчёт из фактических OOF-результатов эксперимента."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--figures-dir', type=Path, required=True)
    return parser.parse_args()


def register_font() -> str:
    candidates = [
        Path(r'C:\Windows\Fonts\arial.ttf'),
        Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),
    ]
    for candidate in candidates:
        if candidate.exists():
            pdfmetrics.registerFont(TTFont('ReportFont', str(candidate)))
            return 'ReportFont'
    raise FileNotFoundError('Не найден шрифт с поддержкой кириллицы (Arial или DejaVu Sans).')


def fmt(value: object, digits: int = 5) -> str:
    if value is None:
        return 'n/a'
    if isinstance(value, (int, float)):
        return f'{value:.{digits}f}'
    return str(value)


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace('\n', '<br/>'), style)


def image_or_note(path: Path, note: str, width: float, styles: dict[str, ParagraphStyle]):
    if path.exists():
        image = Image(str(path))
        ratio = image.imageHeight / image.imageWidth
        image.drawWidth = width
        image.drawHeight = min(width * ratio, 11.2 * cm)
        return image
    return paragraph(note, styles['BodyText'])


def page_number(canvas, document) -> None:
    canvas.saveState()
    canvas.setFont('ReportFont', 8)
    canvas.setFillColor(colors.HexColor('#666666'))
    canvas.drawString(1.8 * cm, 1.1 * cm, 'Home Credit Default Risk - учебный ML-проект')
    canvas.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, f'Страница {document.page}')
    canvas.restoreState()


def make_table(data: list[list[str]], widths: list[float]) -> Table:
    table = Table(data, colWidths=widths, repeatRows=1, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F4E78')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, -1), 'ReportFont'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('LEADING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.HexColor('#B7C9D6')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#EAF2F8')]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    return table


def build(payload: dict, output: Path, figures: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    font = register_font()
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='TitleRu', parent=styles['Title'], fontName=font, fontSize=23, leading=29, alignment=TA_CENTER, textColor=colors.HexColor('#17365D')))
    styles.add(ParagraphStyle(name='HeadingRu', parent=styles['Heading1'], fontName=font, fontSize=16, leading=20, spaceAfter=8, textColor=colors.HexColor('#1F4E78')))
    styles.add(ParagraphStyle(name='SubRu', parent=styles['Heading2'], fontName=font, fontSize=11, leading=14, spaceBefore=5, spaceAfter=4, textColor=colors.HexColor('#1F4E78')))
    styles['BodyText'].fontName = font
    styles['BodyText'].fontSize = 9.5
    styles['BodyText'].leading = 13
    styles['BodyText'].spaceAfter = 6
    document = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=1.8 * cm, leftMargin=1.8 * cm, topMargin=1.6 * cm, bottomMargin=1.8 * cm)
    story = []
    width = A4[0] - 3.6 * cm

    comparison = payload['comparison']
    best = max(comparison, key=lambda row: row.get('roc_auc') or -1)
    blend_weights = ', '.join(f'{name}: {weight:.0%}' for name, weight in payload['blend_weights'].items())

    # Page 1 - title and answer-first summary.
    story += [Spacer(1, 2.0 * cm), paragraph('Кредитный скоринг: Home Credit Default Risk', styles['TitleRu']), Spacer(1, 0.45 * cm)]
    story.append(paragraph('Итоговый отчёт по учебному проекту машинного обучения', styles['HeadingRu']))
    story.append(paragraph(
        f"<b>Задача.</b> Предсказать вероятность дефолта по кредитной заявке. Основная метрика - ROC-AUC. "
        f"В финальном эксперименте использованы {payload['n_splits']}-fold стратифицированная CV, "
        f"{payload['n_train_rows']:,} train-заявок, {payload['n_test_rows']:,} test-заявок и {payload['n_features']:,} признаков.", styles['BodyText']))
    story.append(paragraph(
        f"<b>Главный результат.</b> Лучшая OOF-модель - {best['model']}: ROC-AUC {fmt(best.get('roc_auc'))}, "
        f"PR-AUC {fmt(best.get('pr_auc'))}. Веса финального blend: {blend_weights}.", styles['BodyText']))
    story.append(paragraph(
        '<b>Принцип валидности.</b> Все признаки строятся без TARGET; внешние таблицы агрегируются до SK_ID_CURR; '
        'все fit-преобразования и подбор параметров выполняются внутри train-части CV-фолда. Test не используется для оценки качества.', styles['BodyText']))
    story += [Spacer(1, 0.4 * cm), paragraph('Что сделано', styles['SubRu'])]
    for item in [
        'EDA, профилирование пропусков, аномалий, классового дисбаланса и покрытия внешних источников.',
        'Feature engineering по анкете, бюро, прежним заявкам, POS, платежам и кредитным картам.',
        'Сравнение классических моделей, LightGBM, CatBoost, регуляризованной MLP и OOF-blend.',
        'SHAP-интерпретация, fairness-диагностика и готовый Kaggle submission.',
    ]:
        story.append(paragraph(f'• {item}', styles['BodyText']))
    story.append(PageBreak())

    # Page 2 - EDA and source coverage.
    story += [paragraph('1. Данные и EDA', styles['HeadingRu'])]
    coverage = [['Источник', 'Покрытие train', 'Строк на клиента']]
    for row in payload['source_coverage']:
        coverage.append([row['source'], f"{100 * row['coverage_train_clients']:.2f}%", f"{row['mean_rows_per_client']:.1f}"])
    story.append(make_table(coverage, [7.6 * cm, 4.0 * cm, 4.0 * cm]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(paragraph(
        'Источники имеют разную продуктовую разреженность. Отсутствие записи оставлено как NaN: это может означать отсутствие соответствующей истории, а не ошибку данных. '
        'Проверка ссылочной целостности подтверждает, что внешние ID принадлежат train/test заявкам.', styles['BodyText']))
    story.append(image_or_note(figures / '01_target_and_age.png', 'График баланса классов и возраста не найден.', width, styles))
    story.append(PageBreak())

    # Page 3 - feature engineering and audit.
    story += [paragraph('2. Признаки и защита от утечки', styles['HeadingRu'])]
    story.append(paragraph(
        'Сформированы отношения дохода, кредита и аннуитета; missing-indicators; агрегаты сумм, средних, максимумов, минимумов и дисперсий; доли статусов; '
        'а также недавние срезы кредитной истории. Это позволяет модели отличать нагрузку, просрочки, глубину и давность истории.', styles['BodyText']))
    audit_rows = [['Проверка', 'Статус']]
    for row in payload['leakage_audit']:
        audit_rows.append([row['проверка'], 'пройдена' if row['пройдена'] else 'не пройдена'])
    story.append(make_table(audit_rows, [12.1 * cm, 3.5 * cm]))
    story.append(Spacer(1, 0.25 * cm))
    build_rows = [['Источник', 'Новых признаков', 'Время, c']]
    for row in payload['feature_build_audit']:
        build_rows.append([row['source'], str(row['new_features']), f"{row['seconds']:.1f}"])
    story.append(make_table(build_rows, [7.5 * cm, 4.0 * cm, 4.1 * cm]))
    story.append(paragraph(
        'Важное ограничение: временная допустимость исторических таблиц опирается на описание Kaggle, где эти записи относятся к предыдущим кредитам. '
        'Проверки кода и матрицы исключают техническую утечку, но не превращают конкурсные данные в причинное исследование.', styles['BodyText']))
    story.append(PageBreak())

    # Page 4 - model comparison.
    story += [paragraph('3. Модели и OOF-валидация', styles['HeadingRu'])]
    metrics = [['Модель', 'ROC-AUC', 'PR-AUC', 'Brier', 'F1', 'Время, c']]
    for row in sorted(comparison, key=lambda x: x.get('roc_auc') or -1, reverse=True):
        metrics.append([str(row['model']), fmt(row.get('roc_auc')), fmt(row.get('pr_auc')), fmt(row.get('brier')), fmt(row.get('f1')), f"{row.get('seconds', 0):.1f}"])
    story.append(make_table(metrics, [4.4 * cm, 2.0 * cm, 2.0 * cm, 1.8 * cm, 1.6 * cm, 2.0 * cm]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(paragraph(
        'Качество сравнивается по out-of-fold прогнозам, поэтому каждое наблюдение оценено моделью, которая не видела его label. '
        'Вес blend выбирался исключительно по OOF ROC-AUC, а не по public leaderboard.', styles['BodyText']))
    story.append(image_or_note(figures / '06_mlp_learning_curves.png', 'Кривые обучения MLP не найдены.', width, styles))
    story.append(PageBreak())

    # Page 5 - interpretation and takeaways.
    story += [paragraph('4. Интерпретация, fairness и выводы', styles['HeadingRu'])]
    story.append(image_or_note(figures / '07_shap_summary.png', 'SHAP-график не найден.', width, styles))
    fairness = payload.get('fairness', [])
    if fairness:
        story.append(Spacer(1, 0.2 * cm))
        fair_rows = [['Сегмент', 'Группа', 'n', 'ROC-AUC', 'Brier']]
        for row in fairness[:10]:
            fair_rows.append([row['dimension'], str(row['group']), str(row['n']), fmt(row.get('roc_auc')), fmt(row.get('brier'))])
        story.append(make_table(fair_rows, [3.1 * cm, 3.0 * cm, 2.2 * cm, 3.5 * cm, 3.5 * cm]))
    story.append(Spacer(1, 0.2 * cm))
    story.append(paragraph(
        f"<b>Итог.</b> Для submission использован OOF-blend с ROC-AUC {fmt(best.get('roc_auc'))}. "
        'Результат следует трактовать как качество ранжирования на конкурсной постановке. Перед прикладным использованием необходимы отдельные проверки временной стабильности, калибровки, юридических ограничений и fairness по релевантным защищаемым группам.', styles['BodyText']))

    document.build(story, onFirstPage=page_number, onLaterPages=page_number)
    pages = len(PdfReader(str(output)).pages)
    if not 3 <= pages <= 8:
        raise RuntimeError(f'PDF должен занимать 3-8 страниц, получено: {pages}')
    print(f'PDF_REPORT_OK pages={pages} path={output}')


if __name__ == '__main__':
    args = parse_args()
    build(json.loads(args.results.read_text(encoding='utf-8')), args.output, args.figures_dir)
