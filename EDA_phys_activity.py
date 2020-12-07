import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime as dt
from datetime import timedelta as td
from matplotlib import dates


def resample_data():
    """
    Исходные датасеты содержат данные показаний акселерометра с частотой записи 100 Гц.
    Функция осуществляет ресэмпл до частоты 10 Гц с усреднением измерений.
     """

    df1 = pd.read_csv('01-0.csv')
    df2 = pd.read_csv('02-0.csv')
    df1.dt = pd.to_datetime(df1.dt)
    df2.dt = pd.to_datetime(df2.dt)
    df1 = df1.set_index('dt', drop=True)
    df2 = df2.set_index('dt', drop=True)
    if df1.index.is_unique and df2.index.is_unique:
        df1_resampled = df1.resample('100L').mean()
        df2_resampled = df2.resample('100L').mean()
        df1_resampled.to_csv('01-0-resampled.csv')
        df2_resampled.to_csv('02-0-resampled.csv')


def load_data():
    """ Загрузка датасетов и индексация временными метками. """

    df1 = pd.read_csv('01-0-resampled.csv')
    df2 = pd.read_csv('02-0-resampled.csv')
    if df1.dt.is_unique and df2.dt.is_unique:
        df1.dt = pd.to_datetime(df1.dt)
        df2.dt = pd.to_datetime(df2.dt)
        df1 = df1.set_index('dt', drop=True)
        df2 = df2.set_index('dt', drop=True)
        return df1, df2
    print("Check datasets for index column uniqueness.")
    return df1, df2


def class_balance_info(df1, df2):
    """ Информация по балансу классов. """

    plt.figure(figsize=(4, 6))
    plt.bar(['01-0','02-0'], [df1.count()[0], df2.count()[0]], color=['r', 'g'])
    plt.xlabel('participants', fontsize=12)
    plt.ylabel('observation count', fontsize=12)
    plt.title('Class balance')
    plt.show()


def show_data_detail(df, start_ind=0, end_ind=101):

    df.iloc[start_ind:end_ind, :3].plot(marker='o', rot=30, grid=True, figsize=(16, 8))
    plt.grid(which='major', linestyle='-.')
    plt.grid(which='minor', linestyle=':')
    plt.legend(fontsize=18)
    plt.xlabel('time', fontsize=14)
    plt.show()


def show_data_in_3d(df1, df2):

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(df1.x.values, df1.y.values, df1.z.values, label='01-0', c='r')
    ax.scatter(df2.x.values, df2.y.values, df2.z.values, label='02-0', c='g')
    plt.legend(fontsize=14)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def show_pairplot(df1, df2):

    df1['participants'], df2['participants'] = '01-0', '02-0'
    df = pd.concat([df1, df2])
    sns.set_style("white")
    sns.pairplot(df, hue='participants', diag_kind="hist")
    plt.show()


def show_boxplots(df1, df2):
    """ Сравнение boxplots обоих участников с выбросами и без выбросов. """

    df1['participants'], df2['participants'] = '01-0', '02-0'
    df = pd.concat([df1, df2])
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 16))
    fig.suptitle('Physical activity. Boxplot comparing', y=0.92, fontsize=20)
    for i, c in enumerate(['x', 'y', 'z']):
        sns.boxplot(x='participants', y=c, data=df, ax=axes[0, i])
        sns.boxplot(x='participants', y=c, data=df, ax=axes[1, i], sym='')


def show_daily_phys_activity(df1, df2):
    """
    Построение попарных сравнительных графиков ежедневной физической активности каждого участника.
    На каждом графике - суточная активность участника, начиная с первого дня наблюдений.
    """

    # Приведем временной диапазон участника 02-0 к диапазону 01-0 для удобства сравнения
    df2 = df2[(df2.index < dt(2019, 9, 7, 13)) & (df2.index > dt(2019, 9, 4, 12, 25))]

    # Данные для вывода на график суточных показаний акселерометра каждого участника
    day_starts1 = [df1.index[0] + td(days=i) for i in range(3)]
    df1_days = [df1[(df1.index >= d) & (df1.index < (d + td(days=1)))] for d in day_starts1]
    day_starts2 = [df2.index[0] + td(days=i) for i in range(3)]
    df2_days = [df2[(df2.index >= d) & (df2.index < (d + td(days=1)))] for d in day_starts2]
    df_daily = []
    for i in range(3):
        df_daily.extend([df1_days[i], df2_days[i]])

    # Построение и вывод графиков
    fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(16, 32))
    fig.suptitle('Daily general picture', y=0.90, fontsize=20)
    for i, ax in enumerate(axes):
        ax.plot(df_daily[i].index, df_daily[i].iloc[:, :3])
        if not i % 2:
            ax.set_title("participant 01-0", loc='left', fontsize=14)
        else:
            ax.set_title("participant 02-0", loc='left', fontsize=14)
        ax.grid(linestyle='-.')
        ax.set_xlim(df_daily[i].index[0], df_daily[i].index[-1])
        ax.minorticks_on()
        ax.grid(which='major', linestyle='-.')
        ax.grid(which='minor', linestyle=':')
        ax.set_ylim(-8,7)
        fmt = dates.DateFormatter("%d-%m %H:%M")
        ax.xaxis.set_major_formatter(fmt)
    plt.legend(('X', 'Y', 'Z'))
    plt.show()


def show_remarkable_fliers(df):

    # Day 1, 06:46:10 - 06:46:40
    df_point1 = df['2019-08-12 06:46:10':'2019-08-12 06:46:40']
    # Day 2, 07:43:15 - 07:44:00
    df_point2 = df['2019-08-13 07:43:15':'2019-08-13 07:44:00']
    # Day 3, 07:27:20 - 07:27:55
    df_point3 = df['2019-08-14 07:27:20':'2019-08-14 07:27:55']

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(16, 12))
    df_point1.plot(ax=axes[0])
    df_point2.plot(ax=axes[1])
    df_point3.plot(ax=axes[2])
    plt.show()
