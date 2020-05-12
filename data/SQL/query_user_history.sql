insert into z_seanchuang.user_history_event
with t0 as (
  select
    targeting_id,
    j(configuration, '$.type') as type,
    json_parse(configuration) as c
  from rds_ad.smartad.targeting
  where
    configuration is not null
),
t1 as (
  select
    targeting_id,
    c2 as c,
    c as p_c,
    j(c2, '$.type') as type
  from (
      select
        targeting_id,
        c,
        case
          when type = 'com.smartnews.ad.dmp.common.targeting.ImmutableUnionAudienceSet' then cast(
            json_extract(c, '$.audienceSet') as array<json>
          )
          when type = 'com.smartnews.ad.dmp.common.targeting.ImmutableIntersectionAudienceSet' then cast(
            json_extract(c, '$.audienceSet') as array<json>
          )
          else array[c]
        end as audience_sets
      from t0
    )
  cross join unnest(audience_sets) as t(c2)
),
t2 as (
  select
    targeting_id,
    c2 as c,
    c as p_c,
    j(c2, '$.type') as type
  from (
      select
        targeting_id,
        c,
        case
          when type = 'com.smartnews.ad.dmp.common.targeting.ImmutableUnionAudienceSet' then cast(
            json_extract(c, '$.audienceSet') as array<json>
          )
          when type = 'com.smartnews.ad.dmp.common.targeting.ImmutableIntersectionAudienceSet' then cast(
            json_extract(c, '$.audienceSet') as array<json>
          )
          else array[c]
        end as audience_sets
      from t1
      where
        type != 'com.smartnews.ad.dmp.common.targeting.ImmutableIntersectionAudienceSet'
    )
  cross join unnest(audience_sets) as t(c2)
),
p as (
  select
    j(p_c, '$.value') as url_pattern,
    cast(
      regexp_extract(j(c, '$.timePeriod'), '(\d+)') as bigint
    ) as time_period,
    targeting_id,
    pixel_ids,
    c
  from (
      select
        c,
        targeting_id,
        cast(
          json_extract(c, '$.pixelIds') as array<varchar>
        ) as pixel_ids,
        cast(
          json_extract(c, '$.rule.value') as array<json>
        ) as p_values
      from t2
      where
        type = 'com.smartnews.ad.dmp.common.targeting.ImmutablePixelEventAudienceSet'
    )
  cross join unnest(p_values) as t(p_c)
  where
    j(p_c, '$.operator') = 'i_contains'
),
p2 as (
  select
    p.url_pattern,
    time_period,
    targeting_id,
    pixel_id
  from p
  cross join unnest (pixel_ids) as t(pixel_id)
),
active_dmp_pixel as (
  select
    p2.targeting_id,
    dp.uid,
    dp.ts
  from ml.tmp_sampled_dmp_pixel dp
  inner join p2
    on dp.dt = '${dt}'
    and date_diff(
      'day', from_unixtime(dp.ts), date('${dt}')
    ) <= p2.time_period
    and strpos(dp.url, p2.url_pattern) > 0
    and dp.key = p2.pixel_id
),
cv as (
  select
    cast(
      reduce(
        sequence(length(tag_id), 1),
        0,
        (s, i) -> s + if(
          codepoint(cast(substr(tag_id, length(tag_id) - i + 1, 1) as varchar(1)))
          < codepoint('a'),
          codepoint(cast(substr(tag_id, length(tag_id) - i + 1, 1) as varchar(1)))
          - codepoint('0'),
          codepoint(cast(substr(tag_id, length(tag_id) - i + 1, 1) as varchar(1)))
          - codepoint('a') + 10
        ) * pow(36, i - 1),
        s -> s
      ) as bigint
    ) as advertiser_id,
    guid,
    ts
  from ml.tmp_sampled_nginx_beacons
  where
    dt = '${dt}'
),
web_postbacks as (
  select distinct
    guid as ad_id_plus,
    'cvw:' || cast(advertiser_id as varchar) as url
  from cv
),
app_postbacks as (
  select distinct
    aud.ad_id_plus,
    'cva:' || cast(c.advertiser_id as varchar) as url
  from hive_ad.default.ad_audience_v2 aud
  inner join hive_ad.ml.ad_result_v3 ar
    on aud.dt = 'latest'
    and ar.dt > date_format(date_add('day', -30, date('${dt}')),'%Y-%m-%d')
    and ar.dt <= '${dt}'
    and ar.postback > 0
    and aud.vimp_30 >= 1
    and aud.ad_id_plus = ar.ad_id_plus
  inner join rds_ad.smartad.campaign c
    on c.click_action = 'appstore'
    and ar.id = c.campaign_id
),
custom_audience as (
  select distinct
    aud.ad_id_plus,
    'cus:' || cast(targeting_id as varchar) as url
  from upload.ad_s3idlist_audience i
  inner join default.ad_audience_v2 aud
    on i.dt > date_format(date_add('day', -120, date('${dt}')), '%Y-%m-%d')
    and i.dt <= '${dt}'
    and aud.dt = 'latest'
    and aud.vimp_30 >= 1
    and i.ad_id = aud.ad_id_plus
),
pixel_events as (
  select
    uid as ad_id_plus,
    'pix:' || cast(targeting_id as varchar) as url,
    max(ts) as ts
  from active_dmp_pixel
  group by 1, 2
),
app_events as (
  select
    uid as ad_id_plus,
    'app:' || event || '/' || key as url,
    max(ts) as ts
  from ml.tmp_sampled_dmp_app
  where
    dt = '${dt}'
  group by 1, 2
),
news_events as (
  select
    ad_id_plus,
    'nws:' || url_extract_host(snr.url) as url,
    max(snr.read_ts) as ts
  from hive_ad.ml.sn_result_v2 snr
  inner join hive_ad.default.ad_audience_v2 aud
    on aud.dt = 'latest'
    and snr.dt > date_format(date_add('day', -30, date('${dt}')), '%Y-%m-%d')
    and snr.dt <= '${dt}'
    and aud.vimp_30 >= 1
    and snr.read > 0
    and snr.edition = 'ja_JP'
    and snr.url is not null
    and aud.device_token = snr.device_token
  group by 1, 2
),
ad_id_urls as (
  select
    ad_id_plus,
    array_agg(url order by ts desc) as urls
  from (
      select
        ad_id_plus,
        url,
        cast(to_unixtime(now()) as bigint) as ts
      from web_postbacks
      union all
      select
        ad_id_plus,
        url,
        cast(to_unixtime(now()) as bigint) as ts
      from app_postbacks
      union all
      select
        ad_id_plus,
        url,
        cast(to_unixtime(now()) as bigint) as ts
      from custom_audience
      union all
      select
        *
      from pixel_events
      union all
      select
        *
      from app_events
      union all
      select
        *
      from news_events
    )
  group by
    1
)
select
  cast(ad_id_plus as varchar) || ' ' ||
  array_join(shuffle(slice(urls, 1, 1024)), ' ') as sentence,
  '${dt}' as dt
from ad_id_urls
where
  cardinality(array_distinct(urls)) >= 1
;