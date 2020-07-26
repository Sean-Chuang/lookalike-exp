"""
Since LAL don't have A/B test, and advertisor always turn on/off LAL frequently, 
it is really hard to compare different algorithm performance.
However, we could simple compare only on intersect compaign_id for all of compare period.
The main idea of following query is try to find the same LAL campaign_id in over all days, 
and we could easily check CTR/CVR/CPC/CPA for different algo.
"""

with tid_w_c_per as ( 
	select 
		targeting_id, 
		cast( sum(if(c.click_action = 'webview', 1, 0)) as double ) / count(*) as w_c_per 
	from rds_ad.smartad.campaign_targeting_filter ctf 
	inner join rds_ad.smartad.campaign c using(campaign_id) 
	where ctf.enable = 1 
		and c.enable = 1 
		and c.begin_sec <= to_unixtime(now()) 
		and c.end_sec > to_unixtime(now()) 
	group by 1 
), 
active_campaigns_now as ( 
	select 
		distinct campaign_id 
	from rds_ad.smartad.campaign c 
	inner join rds_ad.smartad.campaign_targeting_filter ctf using(campaign_id) 
	inner join rds_ad.smartad.targeting t using(targeting_id) 
	inner join tid_w_c_per twcp using(targeting_id) 
	where t.enable = 1 
		and c.begin_sec <= to_unixtime(now()) 
		and c.end_sec > to_unixtime(now()) 
		and c.enable = 1 
		and ctf.enable = 1 
		and w_c_per >= 0.9 
		and bitwise_and(t.targeting_subtype, 512) > 0 
), 
active_campaigns1 as (
	select 
		distinct campaign_id
	from action_vimp 
	inner join active_campaigns_now using(campaign_id)
	where dt = '2020-05-01'
),
active_campaigns2 as (
	select 
		distinct campaign_id
	from action_vimp 
	inner join active_campaigns1 using(campaign_id)
	where dt = '2020-05-30'
),
active_campaigns3 as (
	select 
		distinct campaign_id
	from action_vimp 
	inner join active_campaigns2 using(campaign_id)
	where dt = '2020-06-28'
),
active_campaigns as (
	select 
		distinct campaign_id
	from action_vimp 
	inner join active_campaigns3 using(campaign_id)
	where dt = '2020-07-24'
),
t_vimp as ( 
	select 
		date_format( date_add('hour', -hour(from_unixtime(timestamp, 9, 0)) % 24, from_unixtime(timestamp, 9, 0)), '%Y-%m-%d %H:00' ) as time, 
		count(*) as vimp, 
		count(distinct campaign_id) as campaign_id_count
	from action_vimp 
	inner join active_campaigns using(campaign_id) 
	where dt >= '2020-05-01' 
	group by 1 
), t_click as ( 
	select 
		date_format( date_add('hour', -hour(from_unixtime(timestamp, 9, 0)) % 24, from_unixtime(timestamp, 9, 0)), '%Y-%m-%d %H:00' ) as time, 
		count(*) as click 
	from action_click 
	inner join active_campaigns using(campaign_id) 
	where dt >= '2020-05-01' group by 1 
), t_cv as ( 
	select 
		date_format( date_add('hour', -hour(from_unixtime(timestamp, 9, 0)) % 24, from_unixtime(timestamp, 9, 0)), '%Y-%m-%d %H:00' ) as time, 
		count(*) as cv 
	from action_postback 
	inner join active_campaigns using(campaign_id) 
	where dt >= '2020-05-01' group by 1 
) select 
	time, campaign_id_count, vimp, click, cv, round(100.0 * click / vimp, 2) as vctr, round(100.0 * cv / click, 2) as cvr 
from t_vimp 
left join t_click using(time) 
left join t_cv using(time) 
order by 1;


with active_campaigns as ( 
	select 
		dt, 
		campaign_id 
	from z_seanchuang.tmp_daily_lal_campaign_info 
), info_table as ( 
	select 
		ar.dt as dt, 
		count(distinct id) as num_campaign, 
		count(*) as vimp, 
		round(sum(least(social_welfare, 1e5))) as sw, 
		coalesce(sum(sales_e6), 0.0) / 1e6 as sales, 
		coalesce(sum(click), 0.0) as click, 
		coalesce(sum(postback), 0.0) as cv 
	from hive_ad.ml.ad_result_v3 ar 
	inner join common.ad_result_ext e using(ots) 
	inner join active_campaigns ac 
		on ac.campaign_id = ar.id and ar.dt = ac.dt group by 1 
) select *, round(100.0 * click / vimp, 2) as vctr, round(100.0 * cv / click, 2) as cvr, round(sales / click, 2) as cpc, round(sales / cv, 2) as cpa from info_table where dt >= '2020-06-14' order by 1;


with campaign_count as (
	select 
		campaign_id,
		count(*) as count
	from hive_ad.z_mitsuhashi.tmp_lal_performance
	where dt >= '2020-04-20'
	group by 1
	order by 2 desc
),
active_campaigns as ( 
	select 
		campaign_id
	from campaign_count
	where count >= 85
), 
info_table as ( 
	select 
		ar.dt as dt, 
		count(distinct id) as num_campaign, 
		count(*) as vimp, 
		round(sum(least(social_welfare, 1e5))) as sw, 
		coalesce(sum(sales_e6), 0.0) / 1e6 as sales, 
		coalesce(sum(click), 0.0) as click, 
		coalesce(sum(postback), 0.0) as cv 
	from hive_ad.ml.ad_result_v3 ar 
	inner join common.ad_result_ext e using(ots) 
	inner join active_campaigns ac 
		on ac.campaign_id = ar.id group by 1 
) select *, round(100.0 * click / vimp, 2) as vctr, round(100.0 * cv / click, 2) as cvr, round(sales / click, 2) as cpc, round(sales / cv, 2) as cpa from info_table 
where dt >= '2020-04-20' order by 1;



