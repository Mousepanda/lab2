"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    """Выполните загрузку и анализ датасета Titanic.

    Нужно: посчитать пропуски, число пассажиров старше 30 лет, средний возраст
    и долю выживших по классам, а также пять наибольших тарифов по убыванию.
    """
    df = pd.read_csv(data.csv_path)

    # Количество строк в датасете
    row_count = int(len(df))

    # Количество пропусков в каждом столбце
    missing_by_column = df.isna().sum().to_dict()
    missing_by_column = {str(k): int(v) for k, v in missing_by_column.items()}

    # Число пассажиров старше 30 лет (NaN автоматически оценивается как False)
    adults_over_30_count = int((df['Age'] > 30).sum())

    # Средний возраст для каждого значения Pclass
    # groupby().mean() автоматически игнорирует пропуски (NaN)
    mean_age_by_pclass = df.groupby('Pclass')['Age'].mean().to_dict()
    mean_age_by_pclass = {int(k): float(v) for k, v in mean_age_by_pclass.items()}

    # Доля выживших для каждого Pclass
    # Среднее по бинарному столбцу Survived (0/1) даёт долю выживших
    survival_rate_by_pclass = df.groupby('Pclass')['Survived'].mean().to_dict()
    survival_rate_by_pclass = {int(k): float(v) for k, v in survival_rate_by_pclass.items()}

    # Пять наибольших значений Fare в порядке убывания
    highest_fares = df['Fare'].nlargest(5).tolist()
    highest_fares = [float(x) for x in highest_fares]

    return TitanicSummary(
        row_count=row_count,
        missing_by_column=missing_by_column,
        adults_over_30_count=adults_over_30_count,
        mean_age_by_pclass=mean_age_by_pclass,
        survival_rate_by_pclass=survival_rate_by_pclass,
        highest_fares=highest_fares,
    )

if __name__ == "__main__":
    result = analyze_titanic(TitanicInput(csv_path="titanic.csv"))
    print(f"Всего строк: {result.row_count}")
    print(f"Пропуски по столбцам: {result.missing_by_column}")
    print(f"Пассажиров старше 30: {result.adults_over_30_count}")
    print(f"Средний возраст по классам: {result.mean_age_by_pclass}")
    print(f"Доля выживших по классам: {result.survival_rate_by_pclass}")
    print(f"Топ-5 тарифов: {result.highest_fares}")
