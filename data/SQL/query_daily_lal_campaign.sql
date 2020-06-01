delete from z_seanchuang.tmp_daily_lal_campaign_info where dt=cast(current_date as varchar);

insert into z_seanchuang.tmp_daily_lal_campaign_info
with tid_w_c_per as (
  select
    targeting_id,
    cast(
      sum(if(c.click_action = 'webview', 1, 0)) as double
    ) / count(*) as w_c_per
  from rds_ad.smartad.campaign_targeting_filter ctf
  inner join rds_ad.smartad.campaign c using(campaign_id)
  where
    ctf.enable = 1
    and c.enable = 1
    and c.begin_sec <= to_unixtime(now())
    and c.end_sec > to_unixtime(now())
  group by
    1
),
active_campaigns as (
  select
    campaign_id,
    array_join(
      array_agg(
		cast(ctf.filter_type as varchar) ||
		substr(json_extract_scalar(t.configuration, '$.lookAlikePercentage'), 2)
	  ),
      ','
    ) as targeting
  from rds_ad.smartad.campaign c
  inner join rds_ad.smartad.campaign_targeting_filter ctf using(campaign_id)
  inner join rds_ad.smartad.targeting t using(targeting_id)
  inner join tid_w_c_per twcp using(targeting_id)
  where
    t.enable = 1
    and c.begin_sec <= to_unixtime(now())
    and c.end_sec > to_unixtime(now())
    and c.enable = 1
    and ctf.enable = 1
    and w_c_per >= 0.9
    and bitwise_and(t.targeting_subtype, 512) > 0
  group by
    1
)
select
  advertiser_id,
  com.company_name as owner_company,
  campaign_id,
  a.name as advertiser_name,
  c.name as campaign_name,
  cast(click_action as varchar) as click_action,
  case
    when bidding_type = 0
    and optimization_target = 0 then 'Fixed'
    when bidding_type = 1
    and optimization_target = 0 then 'Daily oCPC CPA'
    when bidding_type = 2
    and optimization_target = 1 then 'Daily oCPC Click'
    when bidding_type = 2
    and optimization_target = 2 then 'Daily oCPC CV'
    when bidding_type = 2
    and optimization_target = 4 then 'Daily oCPC CPM'
    when bidding_type = 3 then 'Monthly oCPC CV'
    else 'Error'
  end as bid_type,
  cast(
    round(cast(budget_day_limit_e6 as double) / 1e6) as bigint
  ) as daily_budget,
  cast(
    round(cast(budget_month_limit_e6 as double) / 1e6) as bigint
  ) as monthly_budget,
  targeting,
  'https://partners.smartnews-ads.com/advertiser/campaign/' || cast(campaign_id as varchar) as partners_url,
  'v3_FT' as internal_version,
  cast(current_date as varchar) as dt
from active_campaigns ac
inner join rds_ad.smartad.campaign c using(campaign_id)
inner join rds_ad.smartad.advertiser a using(advertiser_id)
inner join rds_ad.smartad.insertion_order io using(advertiser_id, order_id)
inner join rds_ad.smartad.company com on com.company_id = owner_company_id
