# Пошаговый прогон на Windows

Инструкция для авторской проверки. Всё выполняется в **PowerShell**.
Совокупное машинное время — около 25 минут, из них без присмотра ~20.

Папка проекта далее обозначена `cranebench`. Откройте PowerShell и перейдите
в неё:

```powershell
cd путь\до\cranebench
```

---

## Шаг 0. Python — 2 минуты

```powershell
python --version
```

Нужен **3.10 или новее**. Если команда не найдена или версия ниже, поставьте
Python с python.org и при установке отметьте «Add python.exe to PATH».

Проверьте, что вы в правильной папке:

```powershell
dir
```

Должны быть видны `pyproject.toml`, `cranebench`, `tests`, `tools`,
`examples`, `PAPER_SoftwareX_draft.md`.

---

## Шаг 1. Окружение и тесты — 5 минут

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest -q
```

Если `Activate.ps1` заблокирован политикой выполнения:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

**Что должно получиться:** `25 passed`. Время — около полутора минут.

**Что означает провал.** Тесты проверяют модели, а не сохранённые числа:
планарные уравнения против собранного численно лагранжиана, пространственные и
парные — против независимого символьного вывода, сохранение энергии, спектры
возмущений, сходимость по шагу, совпадение батчевого и скалярного путей. Если
что-то падает у вас, но проходило у нас — это находка, и её место в статье.

Запишите версии:

```powershell
python -c "import sys, numpy, scipy; print(sys.version); print(numpy.__version__, scipy.__version__)"
```

---

## Шаг 2. Четыре планарные кампании — 4 минуты

```powershell
foreach ($c in "calm","reference","dryden","stress") {
    python examples\run_batch_campaign.py $c 500
}
```

Каждая печатает пять строк вида `PD: 500 runs done` и путь к записанному
файлу. Результаты складываются в `run_batch\`.

**Сверьте хеш модуля метрик.** Он записан в каждый ledger:

```powershell
python -c "import json; print(json.load(open('run_batch/reference_ledger.json'))['metric_hash'])"
```

Если он отличается от хеша в приложенных файлах — модуль метрик изменился, и
сравнивать два набора результатов бессмысленно.

---

## Шаг 3. Перенастройка и повторная stress-кампания — 12 минут

Сначала сетка настройки (около 10 минут, 120 прогонов):

```powershell
python tools\retune.py 900
```

Аргумент — бюджет в секундах. 900 хватает с запасом; если процесс прервётся,
просто запустите ту же команду снова — он продолжит с места остановки.
В конце должно быть `tuning complete`.

Затем кампания с новыми коэффициентами:

```powershell
python examples\run_retuned_stress.py 500
```

---

## Шаг 4. Пространственная кампания — 10 минут

```powershell
python examples\run_spatial_campaign.py --n 150
```

Она не векторизована и идёт дольше остальных. В конце должно быть
`merged 150 paired samples -> run_sp2\spatial_paired.npz`.

Если хотите разбить на части, добавьте `--budget 300` (секунды) и запускайте
команду повторно, пока не появится строка про merge.

---

## Шаг 5. Сверка рукописи с данными — 1 минута

```powershell
python tools\verify_manuscript.py
```

**Ожидаемый результат:**

```
cells checked: 65   tolerance: 1.0%

every table cell in the manuscript matches the campaign files.
```

Скрипт разбирает таблицы прямо из `PAPER_SoftwareX_draft.md`, пересчитывает
каждую ячейку из ваших файлов кампаний и падает на любом расхождении свыше 1 %.
Если хотите увидеть, как он работает, испортите намеренно одну цифру в таблице
рукописи и запустите снова — он её назовёт.

Более жёсткий допуск покажет только округления до трёх значащих цифр:

```powershell
python tools\verify_manuscript.py --tol 0.002
```

---

## Шаг 6. Числа, которые скрипт не проверяет — 40 минут глазами

Он проверяет только таблицы. Сверьте вручную, открыв
`PAPER_SoftwareX_draft.md`:

**Таблица 1 в разделе 3.1** — она берётся из тестов, а не из кампаний:

```powershell
pytest tests\test_dynamics.py tests\test_symbolic.py tests\test_wind.py -q -s
```

**Доли цензурирования** (99 % для PD, 100 % для ZVD, 60 % для LQR):

```powershell
python -c "import sys,numpy as np; sys.path.insert(0,'examples'); from summarise_batch import load; d=load('reference'); [print(c, round(float(np.mean(d[c]['settle_time']>=39.999))*100,1)) for c in ('PD','LQR','ZVD','SMC','HSMC')]"
```

**Чувствительность к порогу** (в разделе 3.2: ZVD от 178 до 495 из 500 при
порогах 4.0–7.0°, у остальных 0):

```powershell
python -c "import sys,numpy as np; sys.path.insert(0,'examples'); from summarise_batch import load; d=load('stress'); [print(t, {c:int(np.sum(d[c]['peak_swing']<=t)) for c in ('PD','LQR','ZVD','SMC','HSMC')}) for t in (4.0,4.8,6.0,7.0,10.0)]"
```

**Ранговая корреляция ρ = 0.70** и **счётчики McNemar** (73 дискордантные пары
против HSMC):

```powershell
python -c "import sys,numpy as np; sys.path.insert(0,'examples'); from scipy import stats; from summarise_batch import load; from cranebench.stats import mcnemar; d=load('reference'); O=['PD','LQR','ZVD','SMC','HSMC']; x=[float(np.mean(d[c]['chatter'])) for c in O]; y=[float(np.mean(d[c]['residual_swing'])) for c in O]; print('rho', round(stats.spearmanr(x,y).statistic,3)); [print(c, mcnemar(d['PD']['bound_ok'], d[c]['bound_ok'])) for c in O[1:]]"
```

**Утверждение про рыскание** — что оно одинаково у всех пяти регуляторов:

```powershell
python -c "import numpy as np; z=np.load('run_sp2/spatial_paired.npz'); Y=np.stack([z[f'{c}__peak_yaw'] for c in ('PD','LQR','ZVD','SMC','HSMC')]); print('max spread:', np.max(Y.max(0)-Y.min(0)), 'mean:', round(float(Y.mean()),3))"
```

---

## Шаг 7. Ссылки — полдня

Откройте `docs\reference_check.csv` в Excel. 36 строк, колонка `provenance`
говорит происхождение каждой:

- **26 строк с меткой `[pool]`** — взяты из списков литературы ваших рукописей
  и при подготовке независимо **не перепроверялись**. Именно их декларация
  обязывает вас проверить.
- **9 строк с меткой `[verified]`** — подтверждены по записи издателя при
  подготовке. Проверьте и их: это быстрее, чем решать, каким доверять.
- **1 строка** — самоцитирование ПО, закроется вместе с Zenodo DOI.

По каждой: откройте `https://doi.org/` + значение колонки `doi`, сверьте
авторов, название, журнал, том, выпуск, год и диапазон страниц, поставьте
отметку в `checked_by_author`. Особого внимания заслуживают Huang & Zhu 2021 и
McKay et al. 1979 — они были достроены по Crossref в последний момент.

---

## Что делать, если что-то не сошлось

Не правьте цифру в статье. Сначала выясните, что изменилось:

1. Сравните `metric_hash` в вашем ledger с приложенным. Разные — изменился
   модуль метрик.
2. Сравните `design_seed` и `wind_seeds`. Разные — вы прогнали другой план.
3. Сравните версии NumPy и SciPy. Расхождение в четвёртом знаке при разных
   версиях BLAS — ожидаемо и как раз является ответом на вопрос о
   кроссплатформенной воспроизводимости, который в статье пока открыт.

Любое расхождение свыше 1 % при совпадающих хеше и сидах — это дефект, и его
надо разбирать, а не сглаживать.
