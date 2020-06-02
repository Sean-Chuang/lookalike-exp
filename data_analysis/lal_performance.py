#!/usr/bin/env python3
import prestodb
from collections import defaultdict
from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

conn=prestodb.dbapi.connect(
    host='localhost',
    port=8081,
    user='chunghsiang.chuang',
    catalog='hive_ad',
    schema='default',
)
cur = conn.cursor()

def save_obj(obj, name):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def query_all_campaign(begin_date):
    cur.execute("with active_campaigns as ( select date_format(date_add('day', -2, date(dt)),'%Y-%m-%d') as dt, campaign_id, campaign_name, bid_type from z_mitsuhashi.tmp_lal_performance ), info_table as ( select ac.campaign_id as campaign_id, ar.dt as dt, campaign_name, bid_type, count(distinct id) as num_campaign, count(*) as vimp, coalesce(sum(sales_e6), 0.0) / 1e6 as sales, coalesce(sum(click), 0.0) as click, coalesce(sum(postback), 0.0) as cv from hive_ad.ml.ad_result_v3 ar inner join active_campaigns ac on ac.campaign_id = ar.id and ar.dt = ac.dt group by 1, 2, 3, 4 ) select *, round(100.0 * click / vimp, 2) as vctr, round(100.0 * cv / click, 2) as cvr, round(sales / click, 2) as cpc, round(sales / cv, 2) as cpa from info_table where dt >= '{}' order by campaign_id, dt".format(begin_date))
    rows = cur.fetchall()
    save_obj(rows, begin_date)


def analysis(data):
    campaign_info = set()
    campaign_data = dict()
    # vimp, sales, click, cv, vctr, cvr, cpc, cpa
    for row in tqdm(data):
        campaign_id, dt, campaign_name, bid_type, num_campaign, vimp, sales, click, cv, vctr, cvr, cpc, cpa = row
        campaign_info.add((campaign_id, campaign_name, bid_type))
        dt = '-'.join(dt.split('-')[1:])
        if campaign_id not in campaign_data:
            campaign_data[campaign_id] = {}
        if dt not in campaign_data[campaign_id]:
            campaign_data[campaign_id][dt] = {'vimp': vimp, 'sales': sales, 'click': click, 'cv':cv, 'vctr':vctr, 'cvr':cvr, 'cpc':cpc, 'cpa':cpa if cpa != 'Infinity' else None}


    for campaign_id in campaign_data:
        df = pd.DataFrame.from_dict(campaign_data[campaign_id], orient='index',
                            columns=['vimp', 'sales', 'click', 'cv', 'vctr', 'cvr', 'cpc', 'cpa'])
        print(df)
        # filled_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
        # sns.lineplot(data=df, markers=filled_markers, dashes=False, linewidth=2.5)
        fig, axs = plt.subplots(2,2)
        sns.lineplot(data=df[['vimp', 'sales']], palette="tab10", linewidth=2.5, ax=axs[0,0])
        sns.lineplot(data=df[['click', 'cpa']], palette="tab10", linewidth=2.5, ax=axs[0,1])
        sns.lineplot(data=df[['cpc', 'cv']], palette="tab10", linewidth=2.5, ax=axs[1,0])
        sns.lineplot(data=df[['vctr', 'cvr']], palette="tab10", linewidth=2.5, ax=axs[1,1])
        for ax in fig.axes:
            # plt.xticks(rotation=45, horizontalalignment='right', fontweight='light')
            plt.setp(ax.get_xticklabels(), rotation=45)

        plt.show()
        break


if __name__ == '__main__':
    b_data = '2020-05-10'
    # data = query_all_campaign(b_data)
    data = load_obj(b_data)
    analysis(data)