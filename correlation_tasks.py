"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations
import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt.

    Разделите наблюдения по полу и для каждой группы вычислите корреляции
    признаков FSIQ, VIQ, PIQ, Weight, Height с MRI_Count методом Пирсона.
    В strongest_mri_feature верните название признака с наибольшим модулем
    корреляции с MRI_Count среди объединённых результатов двух групп.
    """
     # Загружаем таблицу с разделителем tab, корректно обрабатываем 'NA' как пропуски
    df = pd.read_csv(data.csv_path, sep='\t', na_values='NA')

    features = ['FSIQ', 'VIQ', 'PIQ', 'Weight', 'Height']

    men_df = df[df['Gender'] == 'Male']
    women_df = df[df['Gender'] == 'Female']

    men_count = int(len(men_df))
    women_count = int(len(women_df))

    # Корреляции Пирсона для мужчин
    men_mri_correlation = {}
    for feat in features:
        corr = men_df[feat].corr(men_df['MRI_Count'], method='pearson')
        men_mri_correlation[feat] = float(corr) if not pd.isna(corr) else 0.0

    # Корреляции Пирсона для женщин
    women_mri_correlation = {}
    for feat in features:
        corr = women_df[feat].corr(women_df['MRI_Count'], method='pearson')
        women_mri_correlation[feat] = float(corr) if not pd.isna(corr) else 0.0

    # Находим признак с максимальным модулем корреляции среди обеих групп
    max_abs_corr = -1.0
    strongest_mri_feature = ""
    for feat in features:
        current_max = max(abs(men_mri_correlation[feat]), abs(women_mri_correlation[feat]))
        if current_max > max_abs_corr:
            max_abs_corr = current_max
            strongest_mri_feature = feat

    return BrainCorrelationSummary(
        men_count=men_count,
        women_count=women_count,
        women_mri_correlation=women_mri_correlation,
        men_mri_correlation=men_mri_correlation,
        strongest_mri_feature=strongest_mri_feature,
    )

if __name__ == "__main__":
    result = analyze_brain_correlations(BrainDataInput(csv_path="brainsize.txt"))
    print(f"Кол-во мужчин: {result.men_count}")
    print(f"Кол-во женщин: {result.women_count}")
    print(f"Корреляция mri для женщин : {result.women_mri_correlation}")
    print(f"Корреляция mri для мужчин: {result.men_mri_correlation}")
    print(f"Наибольший mri признак: {result.strongest_mri_feature}")
